"""
Provides a Process class that creates a logical tree of processes, indicating which processes need
to wait for other processes finish before they can start. It will run the whole tree as efficiently
as possible, with many processes at once if able.

The exact ordering of starting/completing processes cannot be guarnteed, only that processes that
depend on other processes will not start until the processes they depend on are completed. The idea
is to create all the processes with one 'master' process that you call run() on.
"""

from multiprocessing import cpu_count

class Process:
    __MaxAtOnce = cpu_count()
    __semaphore = None

    @staticmethod
    def set_max_count(x):
        """
        Sets the maximum number of processes that should be run simultaneously when run() is called.
        """
        if Process.__MaxAtOnce != x:
            if Process.__semaphore != None:
                raise ValueError('Cannot change process counts while a process tree is running')
            Process.__MaxAtOnce = x
    
    @staticmethod
    def get_max_count():
        """
        Gets the maximum number of processes that should be run simultaneously. Defaults to the number of processors.
        """
        return Process.__MaxAtOnce

    def __init__(self, cmd, dependencies = (), cwd = None):
        """
        Creates a new process with other processes as dependencies.
        The cmd should be an array of command line parts.
        The process does not run until run() is called on this process or a process that depends on this process.
        """
        self.cmd = cmd
        self.cmd = ['./test.py', str(self)] #./test.py
        self.cwd = cwd
        self._init_deps(dependencies)
    def _init_deps(self, dependencies):
        self.dependencies = frozenset(dependencies)
        self.parents = set()
        self.proc = None
        self.__is_done = False # done must be marked in a locked region to avoid race conditions
        self.__clear_cached_ancestors()
        for d in self.dependencies:
            d.parents.add(self)
        

    def __repr__(self):
        from pipes import quote
        return "`%s`" % " ".join(quote(str(s)) for s in self.cmd)
    
    def __clear_cached_ancestors(self):
        self.__ancestors = None
        for d in self.dependencies: d.__clear_cached_ancestors()

    def ancestors(self):
        """
        Get all dependency ancestors of this process
        """
        if self.__ancestors == None: 
            ancs = set(self.parents)
            for p in self.parents:
                ancs = ancs.union(p.ancestors())
            self.__ancestors = frozenset(ancs)
        return self.__ancestors

    def __get_all_leaves(self):
        # TODO: this is quite slow
        leaves = set()
        stack = [self]
        while len(stack) > 0:
            x = stack.pop()
            dep = [d for d in x.dependencies if not d.proc]
            if len(dep) == 0:
                leaves.add(x)
            else:
                stack.extend(dep)
        return leaves
    
    def _run(self):
        """
        Starts the process without blocking. At the end self.proc must be set to an
        object with wait() and poll() functions.
        """
        from subprocess import Popen
        self.proc = Popen(self.cmd, cwd=self.cwd) # self.proc must support wait() and poll()
        
    def __run(self):
        try:
            from heapq import heappush
            
            # Run the process, wait for it to finish, and check its error code
            try:
                # TODO: EDP on Linux deadlocks at random times...
                self._run()
                err = self.proc.wait()
            except BaseException as e:
                err = e

            with Process.__conditional:
                self.__is_done = True # done must be marked in a locked region to avoid race conditions
                if err != 0:
                    # Handle error
                    Process.__error = (str(self), str(err))
                else:
                    # Update dependencies
                    for p in self.parents:
                        if not p.__is_done and all((d.__is_done for d in p.dependencies)):
                            heappush(Process.__leaves, (len(p.ancestors()), p))
                Process.__conditional.notify()
        finally:
            # Release the process-count lock
            Process.__semaphore.release()
        
    def run(self):
        """
        Runs this process after running all necessary dependencies. Will not return until the process is finished.
        """

        from threading import BoundedSemaphore, Condition, Thread 
        from heapq import heapify, heappop

        # Checks
        if self.proc != None: raise ValueError('Process already started')
        if Process.__semaphore != None: raise ValueError('Another process tree is already running')
        Process.__error = None

        # Create locks
        Process.__semaphore = BoundedSemaphore(cpu_count()) # for limiting number of processes
        Process.__conditional = Condition() # for locking access to leaves and error

        # Calcualte starting set of leaves
        Process.__leaves = [(len(l.ancestors()), l) for l in self.__get_all_leaves()]
        heapify(Process.__leaves)

        # Keep running processes in the tree until we have completed the root (which is self)
        min = None
        while min != self:
            # Get next process
            with Process.__conditional:
                while len(Process.__leaves) == 0 and not Process.__error:
                    Process.__conditional.wait(60) # wait until we have some available
                Process.__errcheck()
                min = heappop(Process.__leaves)[1]

            # Run it
            Process.__semaphore.acquire() # make sure not too many things are already running
            
            t = Thread(target=min.__run)
            t.daemon = True
            t.start()

        # Wait for the root process to finish
        with Process.__conditional:
            while self.proc == None and not Process.__error: Process.__conditional.wait()
            Process.__errcheck()
        self.proc.wait()
        Process.__errcheck()

        # Cleanup
        del Process.__error
        del Process.__conditional
        del Process.__leaves
        Process.__semaphore = None

    @staticmethod
    def __errcheck():
        from sys import stderr, exit
        if Process.__error:
            print >> stderr, "Process %s resulted in error %s" % Process.__error
            exit(3)
    
    def is_waiting(self):
        """
        True when the process has not been started and is possibly waiting for dependents or in the queue
        """
        return self.proc == None
    def is_ready(self):
        """
        True when the process can be run immediately, meaning all dependents are done
        """
        return self.proc == None and all((d.__is_done for d in self.dependencies))
    def is_running(self):
        """
        True when the process is actively running
        """
        return self.proc and self.proc.poll() == None
    def is_done(self):
        """
        True when the process has finished running and other processes dependent on it can be run
        """
        return self.__is_done # proc and self.proc.poll() != None # done must be marked in a locked region to avoid race conditions
    def mark_as_done(self, inc_dependents = True):
        if self.proc != None: raise ValueError('Process already started')
        if Process.__semaphore != None: raise ValueError('Cannot mark a process as done when a process tree is already running')
        self.__is_done = True
        if inc_dependents:
            for d in self.dependents: d.mark_as_done(inc_dependents)

class PythonProcess(Process):
    class Popen:
        def __init__(self, target, args=(), kwargs={}):
            from  multiprocessing import Process
            self.p = Process(target=target, args=args, kwargs=kwargs)
            self.p.daemon = True
            self.p.start()
        def wait(self): self.p.join(); return self.p.exitcode;
        def poll(self): return self.p.exitcode
    def __init__(self, target, dependencies = (), args=(), kwargs={}):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self._init_deps(dependencies)
    def _run(self):
        self.proc = PythonProcess.Popen(target, args, kwargs)

class PythonThread(Process):
    class Popen():
        def __init__(self, target, args=(), kwargs={}):
            from  threading import Thread
            self.p = Thread(target=target, args=args, kwargs=kwargs)
            self.p.daemon = True
            self.p.start()
        # TODO: add support for return values from threads
        def wait(self): self.p.join(); return 0;
        def poll(self): return 0
    def __init__(self, target, dependencies = (), args=(), kwargs={}):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self._init_deps(dependencies)
    def _run(self):
        self.proc = PythonThread.Popen(target, args, kwargs)
