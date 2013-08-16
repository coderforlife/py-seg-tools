"""
Provides a Tasks class that creates a logical tree of tasks, indicating which tasks need to wait for
other tasks to finish before they can start (based on input and output files). It will run the whole
tree as efficiently as possible, with many processes at once if able.

The exact ordering of starting/completing tasks cannot be guarnteed, only that tasks that depend on
the output of other tasks will not start until the outputing tasks are done.

While tasks are running, the program listens to SIGUSR1. When recieved, the status of the tasks is
output, including the memory load, expected memory load, tasks running, done, and total, and a list
of all tasks that are "ready to go" (have all prerequistes complete but need either task slots or
memory to run).
"""

__all__ = ['Tasks', 'KB', 'MB', 'GB', 'TB']

from abc import ABCMeta, abstractmethod
from functools import total_ordering
#from itertools import chain

from os import getcwd, getpid
from os.path import abspath, exists, getmtime, join, normpath

from heapq import heapify, heappop, heappush

from multiprocessing import cpu_count, Process as PyProcess
from subprocess import check_call, CalledProcessError
from threading import Condition, Thread
from pipes import quote

from calendar import timegm
from time import gmtime, sleep, strftime, strptime, time
import re

from psutil import Process, virtual_memory # not a built-in library

this_proc = Process(getpid())

KB = 1024
MB = 1024*1024
GB = 1024*1024*1024
TB = 1024*1024*1024*1024

@total_ordering
class Task:
    __metaclass__ = ABCMeta
    def __init__(self, name, inputs, outputs, settings, wd = None):
        if len(outputs) == 0: raise ValueError('Each task must output at least one file')
        self.name = name
        self.inputs = frozenset(inputs)
        self.outputs = frozenset(outputs)
        self.settings = frozenset(settings)
        self.wd = abspath(wd) if wd else None
        self.before = set()
        self.after = set()
        self.__all_after = None
        self.done = False
        self.cpu_pressure = 1
        self.mem_pressure = 1*MB
    def __eq__(self, other): return type(self) == type(other) and self.name == self.name
    def __lt__(self, other): return type(self) <  type(other) or  self.name <  self.name
    def __hash__(self):      return hash(self.name+str(type(self)))
    @abstractmethod
    def _run(self):
        """Starts the task and waits, throws exceptions if something goes wrong"""
        pass
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
    def mark_as_done(self):
        """
        Marks a task and all the tasks before it as done. This means when a task tree runs that includes them they will
        not be run.
        """
        self.done = True
        for t in self.before:
            if not t.done: t.mark_as_done()
    def pressure(self, cpu=None, mem=None):
        """
        Gives the task a certain amount of CPU and/or memory pressure (in bytes). The task will not start unless there
        is sufficient memory and CPUs available. The default for a new task is 1 CPU and 1 MB of memory.

        Note: The CPU count is actually a count against "max tasks at once", not actual available CPUs. Max tasks at
        once does default to the total number of CPUs though.

        If a task requires more CPUs then will be allowed, it will only be run when no other tasks are running.
        If a task requries more memory then can be given, it will never run.
        """
        if cpu != None:
            cpu = int(cpu)
            if cpu <= 0: raise ValueError('Number of used CPUs must be positive')
            self.cpu_pressure = cpu
        if mem != None:
            mem = int(mem)
            if mem < 0: raise ValueError('Amount of used memory must be non-negative')
            self.mem_pressure = mem

class TaskUsingProcess(Task):
    def __init__(self, cmd, inputs, outputs, settings, wd):
        self.cmd = cmd
        Task.__init__(self, "`%s`" % " ".join(quote(str(s)) for s in cmd), inputs, outputs, settings, wd)
    def _run(self):
        check_call(self.cmd, cwd=self.wd)
class TaskUsingPythonFunction(Task):
    def __init__(self, target, args, kwargs, inputs, outputs, settings):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        kwargs = ""
        for k,v in self.kwargs: kwargs += ", %s = %s" % (str(k), str(v))
        if len(self.args) == 0: kwargs = kwargs.lstrip(', ')
        Task.__init__(self, "%s(%s%s)" % (self.target.__name__, ", ".join(self.args), kwargs), inputs, outputs, settings)
    # We don't actually need to spawn a thread since there is a thread spawned essentially just for _run()
    def _run(self): self.target(*self.args, **self.kwargs)
class TaskUsingPythonProcess(Task):
    def __init__(self, target, args, kwargs, inputs, outputs, settings, wd):
        self.target = target
        self.args = args
        self.kwargs = kwargs
        kwargs = ""
        for k,v in self.kwargs: kwargs += ", %s = %s" % (str(k), str(v))
        if len(self.args) == 0: kwargs = kwargs.lstrip(', ')
        Task.__init__(self, "%s(%s%s)" % (self.target.__name__, ", ".join(self.args), kwargs), inputs, outputs, settings, wd)
    def _run(self):
        if self.wd:
            def _chdir_first():
                from os import chdir
                chdir(self.wd)
                self.target(*self.args, **self.kwargs)
            p = PyProcess(_chdir_first)
        else:
            p = PyProcess(target=self.target, args=self.args, kwargs=self.kwargs)
        p.daemon = True
        p.start()
        p.join()
        if p.exitcode: raise CalledProcessError(p.exitcode, str(self))

class Tasks:
    __time_format = '%Y-%m-%d %H:%M:%S' # static, constant

    def __init__(self, log, settings={}, max_tasks_at_once=None, workingdir=None):
        self.workingdir = abspath(workingdir) if workingdir else getcwd()
        self.max_tasks_at_once = int(max_tasks_at_once) if max_tasks_at_once else cpu_count()
        self.settings = settings
        self.logname = normpath(join(self.workingdir, log))
        self.outputs     = {} # key is a filename, value is a task that outputs that file
        self.inputs      = {} # key is a filename, value is a list of tasks that need that file as input
        self.stand_alone = set() # list of tasks that have no input files
        self.all_tasks   = {} # all tasks, indexed by their string representation
        self.__conditional = None

    def add(self, cmd, inputs, outputs, settings=(), wd=None):
        """
        Adds a new task.
        The cmd should be an array of command line parts (with the first one being the program to run).
        The task does not run until sometime after run() is called.
        The inputs, outputs, and settings are used to determine dependencies between individual tasks and if individual tasks can be skipped or not
        """
        return self._add(TaskUsingProcess(cmd, inputs, outputs, settings, wd if wd else self.workingdir))
    def add_func(self, target, args, kwargs, inputs, outputs, settings=(), wd=None, seperate_process = True):
        """
        Adds a new task that is a Python function call.
        The target needs to be a callable object (function).
        The args and kwargs are the list of arguments and dictionary of keyword arguments.
        If seperate_process is False then the task is run in the current process, which reduces some overhead but does experience issues with working-directory and the global iterpreter lock.
        The task does not run until sometime after run() is called.
        The inputs, outputs, and settings are used to determine dependencies between individual tasks and if individual tasks can be skipped or not
        """
        if seperate_process:
            t = TaskUsingPythonProcess(target, args, kwargs, inputs, outputs, settings, wd if wd else self.workingdir)
        elif wd:
            raise ValueError('Working-directory cannot be changed except for seperate processes')
        else:
            t = TaskUsingPythonFunction(target, args, kwargs, inputs, outputs, settings)
        return self._add(t)
    def _add(self, task):
        # Processes the input and output information from the task
        # Updates all before and after lists as well
        if not task.inputs.isdisjoint(task.outputs): raise ValueError('A task cannot output a file that it needs for input')
        is_stand_alone = len(task.inputs) == 0
        new_inputs  = task.inputs  | (self.overall_inputs() - task.outputs)
        new_outputs = task.outputs | (self.overall_outputs() - task.inputs)
        if not new_inputs.isdisjoint(new_outputs): raise ValueError('Task addition will cause a cycle in dependencies')
        for o in task.outputs:
            if o in self.outputs: raise ValueError('Each file can only be output by one task')
        for o in task.outputs:
            self.outputs[o] = task
            if o in self.inputs:
                task.after.update(self.inputs[o])
                task._clear_cached_all_after()
                for t in self.inputs[o]: t.before.add(task)
        if is_stand_alone: self.stand_alone.add(task)
        else:
            for i in task.inputs:
                self.inputs.setdefault(i, []).append(task)
                if i in self.outputs:
                    task.before.add(self.outputs[i])
                    self.outputs[i].after.add(task)
                    self.outputs[i]._clear_cached_all_after()
        self.all_tasks[str(task)] = task
        #if (len(self.overall_inputs()) == 0 and len(self.stand_alone) == 0) or len(self.overall_outputs()) == 0: raise ValueError('Tasks are now cyclic')
        return task
    def find(self, cmd): return self.all_tasks.get(cmd)
    def overall_inputs(self):  return self.inputs.viewkeys() - self.outputs.viewkeys() #set(self.inputs.iterkeys()) - set(self.outputs.iterkeys())
    def overall_outputs(self): return self.outputs.viewkeys() - self.inputs.viewkeys() #set(self.outputs.iterkeys()) - set(self.inputs.iterkeys())
    def __check_acyclic(self):
        if len(self.outputs) == 0 and len(self.inputs) == 0: return
        overall_inputs  = {t for f in self.overall_inputs() for t in self.inputs[f]}
        if (len(overall_inputs) == 0 and len(self.stand_alone) == 0) or len(self.overall_outputs()) == 0: raise ValueError('Tasks are cyclic')
        for t in overall_inputs: t.all_after()
        for t in self.stand_alone: t.all_after()

    def display_stats(self, signum = 0, frame = None):
        print '=' * 80
        
        mem_sys = virtual_memory()
        mem_task = Tasks.__get_mem_used()
        mem_press = self.__mem_pressure
        mem_avail = mem_sys.available - max(mem_press - mem_task, 0)
        print 'Memory (GB): System: %d / %d    Tasks: %d [%d], Avail: %d' % (
            (mem_sys.total - mem_sys.available) // GB, mem_sys.total // GB, mem_task // GB, mem_press // GB, mem_avail // GB)

        task_run   = sum(1 for t in self.all_tasks.itervalues() if hasattr(t, 'running') and t.running)
        task_done  = sum(1 for t in self.all_tasks.itervalues() if t.done)
        task_next  = len(self.__next)
        task_total = len(self.all_tasks)
        task_press = self.__cpu_pressure
        task_max = self.max_tasks_at_once
        print 'Tasks:       Running: %d [%d] / %d, Done: %d / %d, Upcoming: %d' % (task_run, task_press, task_max, task_done, task_total, task_next)

        for priority, task in self.__next:
            text = str(task)
            if len(text) > 60: text = text[:56] + '...' + text[-1]
            mem = str(task.mem_pressure // GB) + 'GB' if task.mem_pressure >= GB else ''
            if task.mem_pressure <= mem_avail: mem += '*'
            cpu = str(task.cpu_pressure) + 'x' if task.cpu_pressure >= 2 else ''
            if min(task_max, task.cpu_pressure) <= (task_max - task_press): cpu += '*'
            print '%4d %-60s %5s %4s' % (priority, text, mem, cpu) 

        print '=' * 80
    
    def __run(self, task):
        # Run the task, wait for it to finish
        err = None
        try:
            task.running = True
            task._run() # TODO: EDP on Linux deadlocks at random times...
            del task.running
        except BaseException as e:
            err = e

        with self.__conditional:
            if err: self.__error = err # Handle error
            else:
                task.done = True # done must be marked in a locked region to avoid race conditions
                # Update subsequent tasks
                for t in task.after:
                    if not t.done and all(b.done for b in t.before):
                        heappush(self.__next, (len(self.all_tasks) - len(t.all_after()), t))
                self.__last.discard(task)
                # Log completion
                self.__log.write(strftime(Tasks.__time_format, gmtime(time()+1))+" "+str(task)+" \n") # add one second for slightly more reliability in determing if outputs are legal
            self.__cpu_pressure -= min(self.max_tasks_at_once, task.cpu_pressure)
            self.__mem_pressure -= task.mem_pressure
            self.__conditional.notify()

    @staticmethod
    def __get_mem_used():
        """Gets the memory used by this process and all its children"""
        mem = this_proc.get_memory_info()[0]
        for p in this_proc.get_children(True):
            try:
                if p.is_running():
                    mem += p.get_memory_info()[0]
            except: pass
        return mem

        # This would be nice, but it turns out it crashes the whole program if the process finished between creating the list of children and getting the memory usage
        # Adding "if p.is_running()" would help but still have a window for the process to finish before getting the memory usage
        #return sum((p.get_memory_info()[0] for p in this_proc.get_children(True)), this_proc.get_memory_info()[0])

    def __next_task(self):
        # Must be called while self.__conditional is acquired
        avail_cpu = self.max_tasks_at_once - self.__cpu_pressure
        if len(self.__next) == 0 or avail_cpu == 0: return None
        avail_mem = virtual_memory().available - max(self.__mem_pressure - Tasks.__get_mem_used(), 0)

        # First do a fast to see if the very next task is doable
        # This should be very fast and will commonly be where the process ends
        priority, task = self.__next[0]
        needed_cpu = min(self.max_tasks_at_once, task.cpu_pressure)
        if needed_cpu <= avail_cpu and task.mem_pressure <= avail_mem:
            heappop(self.__next)
            self.__cpu_pressure += needed_cpu
            self.__mem_pressure += task.mem_pressure
            return task

        # Second do a slow check of all upcoming tasks
        # This can be quite slow if the number of upcoming processes is long
        try:
            priority, task, i = min((priority, task, i) for i, (priority, task) in enumerate(self.__next) if min(self.max_tasks_at_once, task.cpu_pressure) <= avail_cpu and task.mem_pressure <= avail_mem)
            self.__next[i] = self.__next.pop() # O(1)
            heapify(self.__next) # O(n) [could be made O(log(N)) with undocumented _siftup/_siftdown]
            self.__cpu_pressure += min(self.max_tasks_at_once, task.cpu_pressure)
            self.__mem_pressure += task.mem_pressure
            return task
        except: pass
    
    def run(self, verbose=False):
        """
        Runs all the tasks in a smart order with many at once. Will not return until all tasks are done.
        
        Setting verbose to True will cause the time and the command to print whenever a new command is about to start.
        """

        # Checks
        if self.__conditional != None: raise ValueError('Tasks already running')
        if len(self.outputs) == 0 and len(self.inputs) == 0 and len(self.stand_alone) == 0: return
        if len(self.outputs) == 0 or (len(self.inputs) == 0 and len(self.stand_alone) == 0): raise ValueError('Invalid set of tasks (likely cyclic)')
        prev_signal = None

        try:
            # Create basic variables and lock
            self.__error = None
            self.__conditional = Condition() # for locking access to Task.done, cpu_pressure, mem_pressure, next, last, log, and error

            # Setup log
            done_tasks = self.__process_log(verbose) if exists(self.logname) else ()
            self.__log = open(self.logname, 'w', 0)
            for k,v in self.settings.iteritems(): self.__log.write("*"+k+"="+str(v)+"\n")
            # TODO: log overall inputs and outputs
            for dc in done_tasks: self.__log.write(dc+"\n")

            # Calcualte set of first and last tasks
            #overall_inputs  = set(chain.from_iterable(self.inputs[f] for f in self.overall_inputs()))
            overall_inputs  = {t for f in self.overall_inputs() for t in self.inputs[f]}
            overall_outputs = {self.outputs[f] for f in self.overall_outputs()}
            if (len(overall_inputs) == 0 and len(self.stand_alone) == 0) or len(overall_outputs) == 0: raise ValueError('Tasks are cyclic')
            for t in overall_inputs: t.all_after() # precompute these (while also checking for cyclic-ness)
            for t in self.stand_alone: t.all_after()
            self.__last = {t for t in overall_outputs if not t.done}
            first = self.stand_alone | overall_inputs
            changed = True
            while changed:
                changed = False
                for t in first.copy():
                    if t.done: first.remove(t); first.update(t.after); changed = True
            self.__next = [(len(self.all_tasks) - len(t.all_after()), t) for t in first if all(b.done for b in t.before)]
            heapify(self.__next)

            self.__cpu_pressure = 0
            self.__mem_pressure = Tasks.__get_mem_used() + 1*MB # assume that the creation of threads and everything will add some extra pressure

            # Set a signal handler
            try:
                from signal import signal, SIGUSR1
                prev_signal = signal(SIGUSR1, self.display_stats)
            except: pass

            # Keep running tasks in the tree until we have completed the root (which is self)
            while len(self.__last) != 0:
                # Get next task
                with self.__conditional:
                    while len(self.__last) > 0 and not self.__error:
                        task = self.__next_task()
                        if task != None: break
                        self.__conditional.wait(15) # wait until we have some available [the timeout is important because otherwise CTRL+C does not allow aborting, also it allows for the case where memory is freed up beyond our control]
                    if len(self.__last) == 0: break
                    if self.__error: raise self.__error

                # Run it
                t = Thread(target=self.__run, args=(task,))
                t.daemon = True
                if verbose: print strftime(Tasks.__time_format) + " Running " + str(task)
                t.start()
                sleep(0) # make sure it starts 

        finally:
            # Cleanup
            if prev_signal: signal(SIGUSR1, prev_signal)
            if hasattr(self, '__log'):
                self.__log.close()
                del self.__log
            if hasattr(self, '__cpu_pressure'): del self.__cpu_pressure
            if hasattr(self, '__mem_pressure'): del self.__mem_pressure
            if hasattr(self, '__next'): del self.__next
            self.__conditional = None
            del self.__error

    def __process_log(self, verbose):
        with open(self.logname, 'r+') as log: lines = [line.strip() for line in log]
        re_date = re.compile('\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d\s')
        lines = [line for line in lines if len(line) != 0]
        #comments = [line for line in lines if line[0] == '#']
        # Note: this will take the last found setting/command with a given and silently drop the others
        settings = {s[0].strip():s[1].strip() for s in (line[1:].split('=',1) for line in lines if line[0] == '*')} # setting => value
        tasks = {line[20:].strip():line[:19] for line in lines if re_date.match(line)} # task string => date/time string
        #if len(lines) != len(comments) + len(settings) + len(commands): raise ValueError('Invalid file format for tasks log')

        # Check Settings
        changed_settings = self.settings.viewkeys() - settings.viewkeys() # new settings
        changed_settings.update(k for k in (self.settings.viewkeys() & settings.viewkeys()) if str(self.settings[k]).strip() != settings[k]) # all previous settings that changed value

        # Check Tasks / Files
        changed = self.all_tasks.viewkeys() - tasks.viewkeys() # new tasks are not done
        for n,dt in tasks.items(): # not iteritems() since we may remove elements
            t = self.find(n)
            if not t: del tasks[n] # task no longer exists
            elif not t.settings.isdisjoint(changed_settings): changed.add(n) # settings changed
            else:
                datetime = timegm(strptime(dt, Tasks.__time_format))
                inputs = (normpath(join(t.wd, f)) for f in t.inputs)
                if any((exists(f) and getmtime(f) >= datetime for f in inputs)) or any(not exists(normpath(join(t.wd, f))) for f in t.outputs):
                    changed.add(n)
        for n in changed.copy(): changed.update(str(t) for t in self.find(n).all_after()) # add every task that comes after a changed task

        # Mark as Done
        done_tasks = tasks.viewkeys() - changed
        for n in done_tasks:
            if verbose: print "Skipping " + n
            self.find(n).done = True
        return sorted((tasks[n] + " " + n) for n in done_tasks)
