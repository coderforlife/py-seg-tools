"""
Provides a Tasks class that creates a logical tree of tasks, indicating which tasks need to wait for
other tasks to finish before they can start (based on input and output files). It will run the whole
tree as efficiently as possible, with many processes at once if able.

The exact ordering of starting/completing tasks cannot be guarnteed, only that tasks that depend on
the output of other tasks will not start until the outputing tasks are done.
"""

from abc import ABCMeta, abstractmethod
from multiprocessing import cpu_count

class Task:
    __metaclass__ = ABCMeta
    def __init__(self, name, inputs, outputs, settings):
        if len(outputs) == 0: raise ValueError('Each task must output at least one file')
        self.name = name
        self.inputs = frozenset(inputs)
        self.outputs = frozenset(outputs)
        self.settings = frozenset(settings)
        self.before = set()
        self.after = set()
        self.__all_after = None
        self.done = False
    def __eq__(self, other): return type(self) == type(other) and self.name == self.name
    def __hash__(self):      return hash(self.name+str(type(self)))
    @abstractmethod
    def _run(self): pass # starts the task and waits, throws exceptions if something goes wrong
    def __repr__(self): return self.name
    def all_after(self, back_stack = set()):
        """Get all tasks that come after this task while performing a test for cycles"""
        if self.__all_after == None:
            if self in back_stack: raise ValueError('Tasks are cyclic')
            back_stack = back_stack.copy()
            back_stack.add(self)
            after = self.after.copy()
            for a in self.after: after.update(a.all_after(), back_stack)
            self.__all_after = frozenset(after)
        return self.__all_after
    def _clear_cached_all_after(self):
        if self.__all_after:
            self.__all_after = None
            for b in self.before: a._clear_cached_all_after()
    def mark_as_done(self, inc_before = True):
        """
        Marks a task and by default the tasks before it as done. This means when a task tree runs that includes them
        they will not be run. Not marking the tasks before as done means that if those tasks are required by some other
        tasks they may still run.
        """
        # TODO: if self.__semaphore != None: raise ValueError('Cannot mark a task as done when tasks are running')
        if inc_before: self.__mark_as_done_recursive()
        else: self.done = True
    def __mark_as_done_recursive(self):
        self.done = True
        for t in self.before:
            if not t.done: t.__mark_as_done_recursive()

class TaskUsingProcess(Task):
    def __init__(self, cmd, inputs, outputs, settings, wd):
        from pipes import quote
        self.cmd = cmd
        self.wd = wd
        Task.__init__(self, "`%s`" % " ".join(quote(str(s)) for s in cmd), inputs, outputs, settings)
    def _run(self):
        from subprocess import check_call
        check_call(self.cmd, cwd=self.wd)
class TaskUsingPythonFunction(Task):
    def __init__(self, target, args, kwargs, inputs, outputs, settings, wd):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.wd = wd
        kwargs = ""
        for k,v in self.kwargs: kwargs += ", %s = %s" % (str(k), str(v))
        if len(self.args) == 0: kwargs = kwargs.lstrip(', ')
        Task.__init__(self, "%s(%s%s)" % (self.target.__name__, ", ".join(self.args), kwargs), inputs, outputs, settings)
    # We don't actually need to spawn a thread since there is a thread spawned essentially just for _run()
    def _run(self): self.target(*self.args, **self.kwargs) # TODO: working directory?
class TaskUsingPythonProcess(Task):
    def __init__(self, target, args, kwargs, inputs, outputs, settings, wd):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        self.wd = wd
        kwargs = ""
        for k,v in self.kwargs: kwargs += ", %s = %s" % (str(k), str(v))
        if len(self.args) == 0: kwargs = kwargs.lstrip(', ')
        Task.__init__(self, "%s(%s%s)" % (self.target.__name__, ", ".join(self.args), kwargs), inputs, outputs, settings)
    def _run(self):
        from  multiprocessing import Process
        from subprocess import CalledProcessError
        p = Process(target=self.target, args=self.args, kwargs=self.kwargs)
        p.daemon = True
        # TODO: working directory?
        p.start()
        p.join()
        if p.exitcode: raise CalledProcessError(p.exitcode, str(self))

class Tasks:
    __MaxAtOnce = cpu_count() # static
    __time_format = '%Y-%m-%d %H:%M:%S' # static, constant

    @staticmethod
    def set_max_at_once(x):
        """
        Sets the maximum number of tasks that can be run simultaneously when run() is called.
        """
        Tasks.__MaxAtOnce = x
    
    @staticmethod
    def get_max_at_once():
        """
        Gets the maximum number of tasks that can be run simultaneously. Defaults to the number of processors.
        """
        return Tasks.__MaxAtOnce

    def __init__(self, log, settings={}, workingdir=None):
        from os import getcwd
        from os.path import join, normpath
        self.workingdir = workingdir if workingdir else getcwd()
        self.settings = settings
        self.logname = normpath(join(self.workingdir, log))
        self.outputs     = {} # key is a filename, value is a task that outputs that file
        self.inputs      = {} # key is a filename, value is a list of tasks that need that file as input
        self.stand_alone = set() # list of tasks that have no input files
        self.all_tasks   = {} # all tasks, indexed by their string representation
        self.__semaphore = None

    def add(self, cmd, inputs, outputs, settings=(), wd=None):
        """
        Adds a new task.
        The cmd should be an array of command line parts (with the first one being the program to run).
        The task does not run until sometime after run() is called.
        The inputs, outputs, and settings are used to determine dependencies between individual tasks and if individual tasks can be skipped or not
        """
        return self._add(TaskUsingProcess(cmd, inputs, outputs, settings, wd if wd else self.workingdir))
    def add_func(self, target, args, kwargs, inputs, outputs, settings=(), wd=None, seperate_process = False):
        """
        Adds a new task that is a Python function call.
        The target needs to be a callable object (function).
        The args and kwargs are the list of arguments and dictionary of keyword arguments.
        If seperate_process is True, 
        The task does not run until sometime after run() is called.
        The inputs, outputs, and settings are used to determine dependencies between individual tasks and if individual tasks can be skipped or not
        """
        _Task = TaskUsingPythonProcess if seperate_process else TaskUsingPythonFunction
        return self._add(_Task(target, args, kwargs, inputs, outputs, settings, wd if wd else self.workingdir))
    def _add(self, task):
        # Processes the input and output information from the task
        # Updates all before and after lists as well
        if not task.inputs.isdisjoint(task.outputs): raise ValueError('A task cannot output a file that it needs for input')
        # TODO: simple checks for acyclic-ness could be added here
        # for example, if after an add the overall_inputs or overall_outputs sets are empty, then we have a cycle
        for o in task.outputs:
            if self.outputs.has_key(o): raise ValueError('Each file can only be output by one task')
        for o in task.outputs:
            self.outputs[o] = task
            if self.inputs.has_key(o):
                task.after.update(self.inputs[o])
                task._clear_cached_all_after()
                for t in self.inputs[o]: t.before.add(task)
        if len(task.inputs) == 0: self.stand_alone.add(task)
        else:
            for i in task.inputs:
                self.inputs.setdefault(i, []).append(task)
                if self.outputs.has_key(i):
                    task.before.add(self.outputs[i])
                    self.outputs[i].after.add(task)
                    self.outputs[i]._clear_cached_all_after()
        #if (len(self.overall_inputs()) == 0 and len(self.stand_alone) == 0) or len(self.overall_outputs()) == 0: raise ValueError('Tasks are cyclic')
        self.all_tasks[str(task)] = task
        return task
    def find(self, cmd): return self.all_tasks.get(cmd)
    def overall_inputs(self):  return set(self.inputs.iterkeys())  - set(self.outputs.iterkeys())
    def overall_outputs(self): return set(self.outputs.iterkeys()) - set(self.inputs.iterkeys())
    def __check_acyclic(self):
        if len(self.outputs) == 0 and len(self.inputs) == 0: return
        overall_inputs = self.overall_inputs()
        overall_outputs = self.overall_outputs()
        if (len(overall_inputs) == 0 and len(self.stand_alone) == 0) or len(overall_outputs) == 0: raise ValueError('Tasks are cyclic')
        for t in overall_inputs: i.all_after()
        for t in self.stand_alone: i.all_after()
    
    def __run(self, task):
        try:
            from heapq import heappush
            from time import strftime, gmtime
            
            # Run the task, wait for it to finish
            err = None
            try:
                # TODO: EDP on Linux deadlocks at random times...
                task._run()
            except BaseException as e:
                err = e

            with self.__conditional:
                if err: self.__error = err # Handle error
                else:
                    task.done = True # done must be marked in a locked region to avoid race conditions
                    # Update subsequent tasks
                    for t in task.after:
                        if not t.done and all((b.done for b in t.before)):
                            heappush(self.__next, (len(t.all_after()), t))
                    self.__last.discard(task)
                    # Log completion
                    self.__log.write(strftime(Tasks.__time_format, gmtime())+" "+str(task)+" \n")
                self.__conditional.notify()
        finally:
            # Release the task-count lock
            self.__semaphore.release()
        
    def run(self, verbose=False):
        """
        Runs all the tasks in a smart order with many at once. Will not return until all tasks are done.
        
        Setting verbose to True will cause the time and the command to print whenever a new command is about to start.
        """

        from threading import BoundedSemaphore, Condition, Thread 
        from os.path import exists
        from heapq import heapify, heappop
        from time import strftime, gmtime

        # Checks
        if self.__semaphore != None: raise ValueError('Tasks already running')
        if len(self.outputs) == 0 and len(self.inputs) == 0: return
        self.__error = None

        try:
            # Create locks
            self.__semaphore = BoundedSemaphore(Tasks.__MaxAtOnce) # for limiting number of tasks
            self.__conditional = Condition() # for locking access to leaves and error

            # Setup log
            if exists(self.logname):
                # TODO: self.__log = self.__process_log(verbose)
                self.__log = open(self.logname, 'w', 0)
            else:
                self.__log = open(self.logname, 'w', 0)
            for k,v in self.settings.iteritems(): self.__log.write("*"+k+"="+str(v)+"\n")
            # TODO: log overall inputs and outputs

            # Calcualte set of first and last tasks
            overall_inputs = self.overall_inputs()
            overall_outputs = self.overall_outputs()
            if (len(overall_inputs) == 0 and len(self.stand_alone) == 0) or len(overall_outputs) == 0: raise ValueError('Tasks are cyclic')
            for t in overall_inputs: i.all_after() # precompute these (while also checking for cyclic-ness)
            for t in self.stand_alone: i.all_after()
            self.__last = set((t for t in (self.outputs[f] for f in overall_outputs) if not t.done))
            first = self.stand_alone.copy()
            for ts in (self.inputs[f] for f in overall_inputs): first.update(ts)
            changed = True
            while changed:
                changed = False
                for t in first.copy():
                    if t.done: first.remove(t); first.update(t.after); changed = True
            self.__next = [(len(t.all_after()), t) for t in first if all(b.done for b in t.before)]
            heapify(self.__next)

            # Keep running tasks in the tree until we have completed the root (which is self)
            while len(self.__last) != 0:
                # Get next task
                with self.__conditional:
                    while len(self.__next) == 0 and len(self.__last) > 0 and not self.__error:
                        self.__conditional.wait(60) # wait until we have some available [the timeout is important because otherwise CTRL+C does not allow aborting]
                    if len(self.__last) == 0: break
                    if self.__error: raise self.__error
                    task = heappop(self.__next)[1]

                # Run it
                self.__semaphore.acquire() # make sure not too many things are already running
                t = Thread(target=self.__run, args=(task,))
                t.daemon = True
                if verbose: print strftime(Tasks.__time_format) + " Running " + str(task)
                t.start()

        finally:
            # Cleanup
            if hasattr(self, '__log'):
                self.__log.close()
                del self.__log
            del self.__error
            if hasattr(self, '__conditional'): del self.__conditional
            if hasattr(self, '__next'):        del self.__next
            self.__semaphore = None
