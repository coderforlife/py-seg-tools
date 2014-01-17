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

On *nix and Windows systems resource usage can be obtained and saved to a log. Each line is first
the name of the task then the rusage fields (see http://docs.python.org/2/library/resource.html#resource-usage
and man 2 getrusage for more information). It will not record Python function tasks that do not run
in a seperate process. On some forms of *nix the ru_maxrss and other fields will always be 0.
"""

# Only the Tasks class along with the byte-size constants are exported
__all__ = ['Tasks', 'KB', 'MB', 'GB', 'TB']

from abc import ABCMeta, abstractmethod
from functools import total_ordering

from os import getcwd, getpid
from os.path import exists, getmtime, join, normpath, realpath

from heapq import heapify, heappop, heappush

from multiprocessing import cpu_count, Process as PyProcess
from subprocess import Popen, CalledProcessError
from threading import Condition, Thread
from pipes import quote

from calendar import timegm
from time import gmtime, sleep, strftime, strptime, time
from datetime import datetime
import re

from psutil import Process, virtual_memory # not a built-in library
try: import saga # not a built-in library, but only required when using clustering
except: pass

this_proc = Process(getpid())

def get_mem_used_by_tree(proc = this_proc):
    """
    Gets the memory used by a process and all its children (RSS). If the process is not
    provided, this process is used. The argument must be a pid or a psutils.Process object.
    Return value is in bytes.
    """
    # This would be nice, but it turns out it crashes the whole program if the process finished between creating the list of children and getting the memory usage
    # Adding "if p.is_running()" would help but still have a window for the process to finish before getting the memory usage
    #return sum((p.get_memory_info()[0] for p in proc.get_children(True)), proc.get_memory_info()[0])
    if isinstance(proc, (int, long)): proc = Process(proc) # was given a PID
    mem = proc.get_memory_info()[0]
    for p in proc.get_children(True):
        try:
            if p.is_running():
                mem += p.get_memory_info()[0]
        except: pass
    return mem

def get_time_used_by_tree(proc = this_proc):
    """
    Gets the CPU time used by a process and all its children (user+sys). If the process is not
    provided, this process is used. The argument must be a pid or a psutils.Process object.
    Return values is in seconds.
    """
    if isinstance(proc, (int, long)): proc = Process(proc) # was given a PID
    time = sum(proc.get_cpu_times())
    for p in proc.get_children(True):
        try:
            if p.is_running():
                time += sum(p.get_cpu_times())
        except: pass
    return time

def write_error(s):
    """
    Writes out an error message to stderr in red text. This is done so that the
    error messages from the Tasks system can be easily distinguished from the
    errors from the underlying commands being run. If we cannot change the text
    color (not supported by OS or redirecting to a file) then just the string is
    written.
    """
    from sys import stderr
    from os import name

    try:    is_tty = stderr.isatty()
    except: is_tty = False
    
    if is_tty:
        if name == "posix": stderr.write("\x1b[1;31m")
        elif name == "nt":
            from ctypes import windll, Structure, c_short, c_ushort, byref
            k32 = windll.kernel32
            handle = k32.GetStdHandle(-12)
            class COORD(Structure):      _fields_ = [("X", c_short), ("Y", c_short)]
            class SMALL_RECT(Structure): _fields_ = [("L", c_short), ("T", c_short), ("R", c_short), ("B", c_short)]
            class CONSOLE_SCREEN_BUFFER_INFO(Structure): _fields_ = [("Size", COORD), ("CursorPos", COORD), ("Color", c_ushort), ("Rect", SMALL_RECT), ("MaxSize", COORD)]
            csbi = CONSOLE_SCREEN_BUFFER_INFO()
            k32.GetConsoleScreenBufferInfo(handle, byref(csbi))
            prev = csbi.Color
            k32.SetConsoleTextAttribute(handle, 12)
    stderr.write(s)
    if is_tty:
        if name == "posix": stderr.write("\x1b[0m")
        elif name == "nt":  k32.SetConsoleTextAttribute(handle, prev)
    stderr.write("\n")


# These constants are for when giving a certain amount of memory pressure to a
# task. So 1 GB can be easily written as 1*GB.
KB = 1024
MB = 1024*1024
GB = 1024*1024*1024
TB = 1024*1024*1024*1024

@total_ordering
class Task:
    """
    Abstract Task class representing a single Task to be run.
    """
    __metaclass__ = ABCMeta
    def __init__(self, parent, name, inputs=(), outputs=(), settings=(), wd=None):
        #if len(outputs) == 0: raise ValueError('Each task must output at least one file')
        self.parent = parent     # the Tasks object that owns this task
        self.name = name         # name of this task
        self.wd = realpath(wd) if wd != None else getcwd() # working directory
        if isinstance(inputs, basestring): inputs = (inputs,)
        if isinstance(outputs, basestring): outputs = (outputs,)
        if isinstance(settings, basestring): settings = (settings,)
        self.inputs = frozenset(realpath(join(self.wd, f)) for f in inputs)
        self.outputs = frozenset(realpath(join(self.wd, f)) for f in outputs)
        self.settings = frozenset(settings)
        self.before = set()       # tasks that come before this task
        self.after = set()        # tasks that come after this task
        self.__all_after = None   # the cache for the all_after function
        self.done = False         # not done yet
        self._cpu_pressure = 1    # default number of CPUs is 1
        self._mem_pressure = 1*MB # default memory pressure is 1 MB
    def __eq__(self, other): return type(self) == type(other) and self.name == other.name
    def __lt__(self, other): return type(self) <  type(other) or  type(self) == type(other) and self.name < other.name
    def __hash__(self):      return hash(self.name+str(type(self)))
    def __repr__(self):      return self.name

    @abstractmethod
    def _run(self):
        """
        Starts the task and waits, throws exceptions if something goes wrong.
        This is in abstract method and is implemented in each derived class.
        """
        pass # abstract method does nothing
    def _run_proc(self, p):
        """
        The _run method for seperate process systems. The argument p is a Popen-like object that has
        the attribute 'pid' and the method 'wait' that takes no arguments and retruns the exit code.
        """
        self.pid = p.pid
        if self.parent._rusagelog:
            from os_ext import wait4
            pid, exitcode, rusage = wait4(self.pid, 0)
            del self.pid
            if exitcode: raise CalledProcessError(exitcode, str(self))
            self.parent._rusagelog.write('%s %f %f %d %d %d %d %d %d %d %d %d %d %d %d %d %d\n' % (str(self), 
                rusage.ru_utime, rusage.ru_stime, rusage.ru_maxrss, rusage.ru_ixrss, rusage.ru_idrss, rusage.ru_isrss,
                rusage.ru_minflt, rusage.ru_majflt, rusage.ru_nswap, rusage.ru_inblock, rusage.ru_oublock,
                rusage.ru_msgsnd, rusage.ru_msgrcv, rusage.ru_nsignals, rusage.ru_nvcsw, rusage.ru_nivcsw))
        else:
            exitcode = p.wait()
            del self.pid
            if exitcode: raise CalledProcessError(exitcode, str(self))
            
    def all_after(self, back_stack = set()):
        """
        Get a set of all tasks that come after this task while performing a test for cycles.
        This can be an expensive operation but is cached so multiple calls to it are fast. The cache
        is cleared after any tasks are added to the tree though.
        """
        if self.__all_after == None:
            if self in back_stack: raise ValueError('Tasks are cyclic')
            back_stack = back_stack.copy()
            back_stack.add(self)
            after = self.after.copy()
            for a in self.after: after.update(a.all_after(), back_stack)
            self.__all_after = frozenset(after)
        return self.__all_after
    def _clear_cached_all_after(self):
        """Clears the cached results of "all_after" recursively."""
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

    @property
    def cpu_pressure(self): return self._cpu_pressure
    @property
    def mem_pressure(self): return self._mem_pressure
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
            self._cpu_pressure = cpu
        if mem != None:
            mem = int(mem)
            if mem < 0: raise ValueError('Amount of used memory must be non-negative')
            self._mem_pressure = mem
    def current_usage(self):
        """
        Gets the current memory (bytes) and total CPU usage (seconds) by this task. Throws exceptions in many cases,
        including if the task does not support this operation.
        """
        p = Process(self.pid)
        return get_mem_used_by_tree(p), get_time_used_by_tree(p)
    def terminate(self): Process(self.pid).terminate()
    def kill(self):      Process(self.pid).kill()

class TaskUsingProcess(Task):
    """
    A single Task that runs using a seperate process. 
    """
    def __init__(self, parent, cmd, inputs=(), outputs=(), settings=(), wd=None, stdin=None, stdout=None, stderr=None):
        """
        Create a new Task using a process. The cmd can either be a command-line
        string or a iterable of command-line parts. The stdin/stdout/stderr can
        be strings (for files), a file-like object, or None for default.
        """
        if isinstance(cmd, basestring):
            import shlex
            cmd = shlex.split(cmd)
        else:
            cmd = [str(a) for a in cmd]
        self.cmd    = cmd
        self.stdin  = stdin
        self.stdout = stdout
        self.stderr = stderr
        Task.__init__(self, parent, "`%s`" % " ".join(quote(str(s)) for s in cmd), inputs, outputs, settings, wd)
    def _run(self):
        stdin  = open(self.stdin,  'r', 1) if isinstance(self.stdin,  basestring) else self.stdin
        stdout = open(self.stdout, 'w', 1) if isinstance(self.stdout, basestring) else self.stdout
        stderr = open(self.stderr, 'w', 1) if isinstance(self.stderr, basestring) else self.stderr
        self._run_proc(Popen(self.cmd, cwd=self.wd, stdin=stdin, stdout=stdout, stderr=stderr))
class TaskUsingCluster(TaskUsingProcess):
    """
    A single Task that runs using a seperate process, either locally or on a cluster.
    THIS IS UNTESTED
    """
    def __init__(self, parent, cmd, inputs=(), outputs=(), settings=(), wd=None, stdin=None, stdout=None, stderr=None):
        # We need copies of the original (relative) intputs and outputs for sending to the server
        self._orig_inputs = frozenset(inputs)
        self._orig_outputs = frozenset(outputs)
        TaskUsingProcess.__init__(self, parent, cmd, inputs, outputs, settings, wd, stdin, stdout, stderr)
    def _run(self):
        if self.parent._cluster:
            # TODO: rusagelog
            # TODO: SGE properties: name queue project

            # Set the command to be run
            desc = saga.job.Description()
            desc.executable = self.cmd[0]
            desc.arguments = self.cmd[1:]
            desc.environment = TODO
            if isinstance(self.stdin, basestring): desc.input = self.stdin
            elif self.stdin != None: raise ValueError("Commands running on a cluster do not support using non-file STDIN")
            if isinstance(self.stdout, basestring): desc.output = self.stdout
            elif self.stdout != None: raise ValueError("Commands running on a cluster do not support using non-file STDOUT")
            if isinstance(self.stderr, basestring): desc.error = self.stderr
            elif self.stderr != None: raise ValueError("Commands running on a cluster do not support using non-file STDERR")
            #desc.working_directory = self.wd # TODO

            # Set the CPU and memory hints
            desc.total_cpu_count = self._cpu_pressure # TODO: determine target's CPU capabilities
            if self._mem_pressure > 1*MB: desc.total_physical_memory = self._mem_pressure / MB

            # Set inputs and outputs
            desc.file_transfer = ([x+" > "+y for x, y in zip(self.inputs,  self._orig_inputs )] +
                                  [x+" < "+y for x, y in zip(self.outputs, self._orig_outputs)])
            desc.cleanup = True

            # TODO: are the stdin/stdout/stderr files copied automatically?

            self.job = self.parent._cluster.service.create_job(desc)
            try:
                self.job.run()
                self.job.wait()
                exitcode = self.job.exit_code
            finally:
                del self.job
            if exitcode: raise CalledProcessError(exitcode, str(self))
        else:
            super(TaskUsingCluster, self)._run()
    @Task.cpu_pressure.getter
    def cpu_pressure(self): return 0 if self.parent._cluster else self._cpu_pressure
    @Task.mem_pressure.getter
    def mem_pressure(self): return 0 if self.parent._cluster else self._mem_pressure
    def current_usage(self):
        if self.parent._cluster:
            return 0, (time() - TaskUsingCluster._get_time(self.job.started) if self.job == saga.job.RUNNING else 0)
        else: return super(TaskUsingCluster, self).current_usage()
    @staticmethod
    def _get_time(x):
        if isinstance(x, (int,long,float)): return x
        elif isinstance(x, basestring):     return strptime(x)
        elif isinstance(x, datetime):       return (x-datetime.utcfromtimestamp(0)).total_seconds()
        else: raise ValueError()
    def terminate(self):
        if hasattr(self, 'job'): self.job.cancel(1)
        else: super(TaskUsingCluster, self).terminate()
    def kill(self):
        if hasattr(self, 'job'): self.job.cancel()
        else: super(TaskUsingCluster, self).kill()
        
class TaskUsingPythonFunction(Task):
    """
    Create a new Task that calls a Python function in the same process.
    THIS IS UNTESTED
    """
    def __init__(self, parent, target, args, kwargs, inputs=(), outputs=(), settings=()):
        """
        The target must be a callable object (like a function). The args and
        kwargs are a tuple and dictionary that are given to the target function
        as the arguments and keyword arguments.
        """
        if not callable(target): raise ValueError('Target is not callable')
        self.target = target
        self.args   = args
        self.kwargs = kwargs
        kwargs = ""
        for k,v in self.kwargs: kwargs += ", %s = %s" % (str(k), str(v))
        if len(self.args) == 0: kwargs = kwargs.lstrip(', ')
        Task.__init__(self, parent, "%s(%s%s)" % (self.target.__name__, ", ".join(self.args), kwargs), inputs, outputs, settings)
    # We don't actually need to spawn a thread since there is a thread spawned essentially just for _run()
    def _run(self): self.target(*self.args, **self.kwargs)
    def current_usage(self): raise NotImplementedError()
    def terminate(self): raise NotImplementedError()
    def kill(self): raise NotImplementedError()

class TaskUsingPythonProcess(Task):
    """
    Create a new Task that calls a Python function in a different process.
    THIS IS UNTESTED
    """
    class Popen:
        """
        This is a Popen-like class for Python multiprocessing processes. It supports the pid
        attribute, the wait() function, and changing the working directory along with standard
        inputs/outputs.
        """
        @staticmethod
        def _get_std(stdxxx, mode):
            from os import fdopen 
            if isinstance(stdxxx, basestring):    return open(stdxxx, mode, 1)
            elif isinstance(stdxxx, (int, long)): return fdopen(stdxxx, mode, 1)
            return stdxxx # assume a file object
        def __init__(target, args, kwargs, wd, stdin, stdout, stderr):
            def _setup_pyproc(target, args, kwargs, wd, stdin, stdout, stderr):
                import sys, os
                from subprocess import STDOUT
                os.chdir(wd)
                if stdin:  sys.stdin  = TaskUsingPythonProcess.Popen._get_std(stdin,  'r')
                if stdout: sys.stdout = TaskUsingPythonProcess.Popen._get_std(stdout, 'w')
                if stderr: sys.stderr = TaskUsingPythonProcess.Popen._get_std(stderr, 'w') if stderr != STDOUT else sys.stdout
                target(*args, **kwargs)
            p = PyProcess(_setup_pyproc, (target, args, kwargs, wd, stdin, stdout, stderr))
            p.daemon = True
            p.start()
            self.proc = p
        @property
        def pid(self): return self.proc.pid
        def wait(self): self.proc.join(); return self.proc.exitcode
    def __init__(self, parent, target, args, kwargs, inputs=(), outputs=(), settings=(), wd=None, stdin=None, stdout=None, stderr=None):
        """
        The target must be a callable object (like a function). The args and
        kwargs are a tuple and dictionary that are given to the target function
        as the arguments and keyword arguments. The stdin/stdout/stderr can be
        strings (for files), a file-like object, or None for default.
        """
        if not callable(target): raise ValueError('Target is not callable')
        self.target = target
        self.args   = args
        self.kwargs = kwargs
        self.stdin  = stdin
        self.stdout = stdout
        self.stderr = stderr
        kwargs = ""
        for k,v in self.kwargs: kwargs += ", %s = %s" % (str(k), str(v))
        if len(self.args) == 0: kwargs = kwargs.lstrip(', ')
        Task.__init__(self, parent, "%s(%s%s)" % (self.target.__name__, ", ".join(self.args), kwargs), inputs, outputs, settings, wd)
    def _run(self):
        self._run_proc(TaskUsingPythonProcess.Popen(self.target, self.args, self.kwargs, self.wd, self.stdin, self.stdout, sys.stderr))

class Tasks:
    """
    Represents a set of tasks that need to be run, possibly with dependencies on
    each other. The tasks are run as efficiently as possible.
    """
    __time_format = '%Y-%m-%d %H:%M:%S' # static, constant

    def __init__(self, log, settings={}, max_tasks_at_once=None, workingdir=None, rusage_log=None):
        """
        Create a new set of Tasks.
          log
            the filepath to a file where to read/write the log of completed
            tasks to
          settings
            the setting names and their values for this run as a dictionary
          max_tasks_at_once
            the maximum number of tasks to run at one time, defaults to the
            number of processors available
          workingdir
            the default working directory for all of the tasks, defaults to the
            current working directory
          rusage_log
            only provide if on *nix - if provided the memory and time usage of
            every task will be logged to the given file
        """
        self.workingdir = realpath(workingdir) if workingdir else getcwd()
        self.max_tasks_at_once = int(max_tasks_at_once) if max_tasks_at_once else cpu_count()
        self.settings = settings
        self.logname = normpath(join(self.workingdir, log))
        self.rusage_log = realpath(join(self.workingdir, rusage_log)) if rusage_log else None
        self.outputs    = {} # key is a filename, value is a task that outputs that file
        self.inputs     = {} # key is a filename, value is a list of tasks that need that file as input
        self.generators = set() # list of tasks that have no input files
        self.cleanups   = set() # list of tasks that have no output files
        self.all_tasks  = {} # all tasks, indexed by their string representation
        self.__conditional = None

    def add(self, cmd, inputs=(), outputs=(), settings=(), can_run_on_cluster=False, wd=None, stdin=None, stdout=None, stderr=None):
        """
        Adds a new task. The task does not run until sometime after run() is called.
          cmd
            an array of command line parts (with the first one being the program to run) or a
            command line string
          inputs, outputs, settings
            lists/tuples of files or names of settings, used to determine dependencies between
            individual tasks and if individual tasks can be skipped or not; each can be an empty
            list implying the task generates files without any file input, performs cleanup tasks,
            or only uses files
          can_run_on_cluster
            set to true if this process should/can run on a cluster if available
          wd
            the working directory of the task, by default it is the working directory of the whole
            set of tasks (which defaults to the current working directory)
          stdin, stdout, stderr
            set the standard input and outputs for the task, can be a file object, file descriptor
            (positive int) or a filename; stderr can also be subprocess.STDOUT if it will output to
            stdout; by default they are the same as the this process
        """
        if can_run_on_cluster:
            t= TaskUsingCluster(self, cmd, inputs, outputs, settings, wd if wd else self.workingdir)
        else:
            t = TaskUsingProcess(self, cmd, inputs, outputs, settings, wd if wd else self.workingdir)
        return self._add(t)
    def add_func(self, target, args=(), kwargs={}, inputs=(), outputs=(), settings=(), seperate_process=True, wd=None, stdin=None, stdout=None, stderr=None):
        """
        Adds a new task that is a Python function call. The task does not run until sometime after
        run() is called.
          target
            a callable object (e.g. a function)
          args, kwargs
            the list of arguments and dictionary of keyword arguments passed to the function
          inputs, outputs, settings
            lists/tuples of files or names of settings, used to determine dependencies between
            individual tasks and if individual tasks can be skipped or not; each can be an empty
            list implying the task generates files without any file input, performs cleanup tasks,
            or only uses files
          seperate_process
            if False then the task is run in the current process, which reduces some overhead but
            does experience issues with working-directory and the global iterpreter lock
          wd (only if seperate_process is False)
            the working directory of the task, by default it is the working directory of the whole
            set of tasks (which defaults to the current working directory)
          stdin, stdout, stderr (only if seperate_process is False)
            set the standard input and outputs for the task, can be a file object, file descriptor
            (positive int) or a filename; stderr can also be subprocess.STDOUT if it will output to
            stdout; by default they are the same as the this process
        """
        if seperate_process:
            t = TaskUsingPythonProcess(self, target, args, kwargs, inputs, outputs, settings, wd if wd else self.workingdir)
        elif wd:
            raise ValueError('Working-directory cannot be changed except for seperate processes')
        else:
            t = TaskUsingPythonFunction(self, target, args, kwargs, inputs, outputs, settings)
        return self._add(t)
    def _add(self, task):
        """
        Actual add function. Checks the task and makes sure the task is valid for this set of tasks.
        Updates the task graph (before and after links for this and other tasks).
        """
        # Processes the input and output information from the task
        # Updates all before and after lists as well
        if len(task.settings - self.settings.viewkeys()) > 0: raise ValueError('Task had settings that were not originally specified')
        if not task.inputs.isdisjoint(task.outputs): raise ValueError('A task cannot output a file that it needs for input')
        # A "generator" task is one with inputs, a "cleanup" task is one with outputs
        is_generator, is_cleanup = len(task.inputs) == 0, len(task.outputs) == 0
        new_inputs  = task.inputs  | (self.overall_inputs() - task.outputs) # input files never seen before
        new_outputs = task.outputs | (self.overall_outputs() - task.inputs) # output files never seen before
        # Check for the creation of a cyclic dependency in tasks
        if not new_inputs.isdisjoint(new_outputs) or \
            (len(new_inputs) == 0 and not is_generator and len(self.generators) == 0) or \
            (len(new_outputs) == 0 and not is_cleanup and len(self.cleanups) == 0): raise ValueError('Task addition will cause a cycle in dependencies')
        # Add the task to the graph
        if is_cleanup: self.cleanups.add(task)
        else:
            for o in task.outputs:
                if o in self.outputs: raise ValueError('Each file can only be output by one task')
            for o in task.outputs:
                self.outputs[o] = task
                if o in self.inputs:
                    task.after.update(self.inputs[o])
                    task._clear_cached_all_after()
                    for t in self.inputs[o]: t.before.add(task)
        if is_generator: self.generators.add(task)
        else:
            for i in task.inputs:
                self.inputs.setdefault(i, []).append(task)
                if i in self.outputs:
                    task.before.add(self.outputs[i])
                    self.outputs[i].after.add(task)
                    self.outputs[i]._clear_cached_all_after()
        self.all_tasks[str(task)] = task
        return task
    def find(self, cmd):
        """Find a task from the string representation of the task."""
        return self.all_tasks.get(cmd)
    def overall_inputs(self):
        """Get the overall inputs required from the entire set of tasks."""
        return self.inputs.viewkeys() - self.outputs.viewkeys() #set(self.inputs.iterkeys()) - set(self.outputs.iterkeys())
    def overall_outputs(self):
        """Get the overall outputs generated from the entire set of tasks."""
        return self.outputs.viewkeys() - self.inputs.viewkeys() #set(self.outputs.iterkeys()) - set(self.inputs.iterkeys())
    def __check_acyclic(self):
        """Run a thorough check for cyclic dependencies. Not actually used anywhere."""
        if len(self.outputs) == 0 and len(self.inputs) == 0: return
        overall_inputs  = {t for f in self.overall_inputs() for t in self.inputs[f]}
        if (len(overall_inputs) == 0 and len(self.generators) == 0) or (len(self.overall_outputs()) == 0 and len(self.cleanups) == 0): raise ValueError('Tasks are cyclic')
        for t in overall_inputs: t.all_after()
        for t in self.generators: t.all_after()

    def display_stats(self, signum = 0, frame = None):
        """
        Writes to standard out a whole bunch of statistics about the current status of the tasks. Do
        not call this except while the tasks are running. It is automatically registered to the USR1
        signal on POSIX systems. The signum and frame arguments are not used but are required to be
        present for the signal handler.
        """
        
        print '=' * 80
        
        mem_sys = virtual_memory()
        mem_task = get_mem_used_by_tree()
        mem_press = self.__mem_pressure
        mem_avail = mem_sys.available - max(mem_press - mem_task, 0)
        print 'Memory (GB): System: %d / %d    Tasks: %d [%d], Avail: %d' % (
            int(round(float(mem_sys.total - mem_sys.available) / GB)),
            int(round(float(mem_sys.total) / GB)),
            int(round(float(mem_task) / GB)),
            int(round(float(mem_press) / GB)),
            int(round(float(mem_avail) / GB)))

        task_done  = sum(1 for t in self.all_tasks.itervalues() if t.done)
        task_next  = len(self.__next)
        task_run   = len(self.__running)
        task_total = len(self.all_tasks)
        task_press = self.__cpu_pressure
        task_max = self.max_tasks_at_once
        print 'Tasks:       Running: %d [%d] / %d, Done: %d / %d, Upcoming: %d' % (task_run, task_press, task_max, task_done, task_total, task_next)

        print '-' * 80
        if task_run == 0:
            print 'Running: none (probably waiting for more memory)'
        else:
            print 'Running:'
            for task in sorted(self.__running):
                text = str(task)
                if len(text) > 60: text = text[:56] + '...' + text[-1]
                real_mem = '? '
                timings = ''
                try:
                    real_mem, t = task.current_usage()
                    real_mem = str(int(round(float(real_mem) / GB)))
                    t = int(round(t))
                    hours, mins, secs = t // (60*60), t // 60, t % 60
                    timing = ('%d:%02d:%02d' % (hours, mins - hours * 60, secs)) if hours > 0 else ('%d:%02d' % (mins, secs))
                except: pass
                mem = str(int(round(float(task.mem_pressure) / GB)))
                print '%-60s %3sGB [%3s] %7s' % (text, real_mem, mem, timing)

        print '-' * 80
        if len(self.__next) == 0:
            print 'Upcoming: none'
        else:
            print 'Upcoming:'
            for priority, task in sorted(self.__next):
                text = str(task)
                if len(text) > 60: text = text[:56] + '...' + text[-1]
                mem = str(int(round(float(task.mem_pressure) / GB))) + 'GB' if task.mem_pressure >= 0.5*GB else ''
                if task.mem_pressure <= mem_avail: mem += '*'
                cpu = str(task.cpu_pressure) + 'x' if task.cpu_pressure >= 2 else ''
                if min(task_max, task.cpu_pressure) <= (task_max - task_press): cpu += '*'
                print '%4d %-60s %5s %4s' % (len(task.all_after()), text, mem, cpu) 

        print '=' * 80
    
    def __run(self, task):
        """
        Actually runs a task. This function is called as a seperate thread. Waits for the task to
        complete and then updates the information about errors, next, last, pressure, and running.
        """
        # Run the task and wait for it to finish
        try: task._run()
        except Exception as e: err = e
        else:                  err = None

        with self.__conditional:
            if not self.__killing:
                if err:
                    write_error("Error in task: " + str(err))
                    self.__error = True
                else:
                    task.done = True # done must be marked in a locked region to avoid race conditions
                    # Update subsequent tasks
                    for t in task.after:
                        if not t.done and all(b.done for b in t.before):
                            heappush(self.__next, (len(self.all_tasks) - len(t.all_after()), t))
                    self.__last.discard(task)
                    # Log completion
                    self.__log.write(strftime(Tasks.__time_format, gmtime(time()+1))+" "+str(task)+" \n") # add one second for slightly more reliability in determing if outputs are legal
            # Remove CPU and memory pressures of this task
            self.__cpu_pressure -= min(self.max_tasks_at_once, task.cpu_pressure)
            self.__mem_pressure -= task.mem_pressure
            # This task is no longer running
            self.__running.remove(task)
            # Notify waiting threads
            self.__conditional.notify()

    def __calc_next(self):
        """
        Calculate the list of tasks that have all prerequisites completed. This also verifies that
        the tasks are truly acyclic (the add() function only does a minimal check). Must be called
        when __next is not locked in another thread (so either before any threads are started or
        self.__conditional is acquired).

        The next list is not returned, but stored in self.__next.

        We recalculate the next list at the very beginning and periodically when going through the
        list of tasks just to make sure the list didn't get corrupted or something.
        """
        first = {t for f in self.overall_inputs() for t in self.inputs[f]}
        first |= self.generators
        if len(first) == 0: raise ValueError('Tasks are cyclic')
        for t in first: t.all_after() # precompute these (while also checking for cyclic-ness)
        changed = True
        while changed:
            changed = False
            for t in first.copy():
                if t.done:
                    first.remove(t)
                    first |= t.after
                    changed = True
        num_tasks = len(self.all_tasks)
        self.__next = [(num_tasks - len(t.all_after()), t) for t in first if all(b.done for b in t.before)]
        heapify(self.__next)

    def __next_task(self):
        """
        Get the next task to be run based on priority and memory/CPU usage. Updates the CPU and
        memory pressures assuming that task will be run.

        Must be called while self.__conditional is acquired.
        """

        if len(self.__next) == 0 and len(self.__running) == 0:
            # Something went wrong... we have nothing running and nothing upcoming... recalulate the next list
            self.__calc_next()
        if len(self.__next) == 0 or self.max_tasks_at_once == self.__cpu_pressure: return None

        # Get available CPU and memory
        avail_cpu = self.max_tasks_at_once - self.__cpu_pressure
        avail_mem = virtual_memory().available - max(self.__mem_pressure - get_mem_used_by_tree(), 0)

        # First do a fast check to see if the very next task is doable
        # This should be very fast and will commonly be where the checking ends
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
            heapify(self.__next) # O(n) [TODO: could be made O(log(N)) with undocumented _siftup/_siftdown]
            self.__cpu_pressure += min(self.max_tasks_at_once, task.cpu_pressure)
            self.__mem_pressure += task.mem_pressure
            return task
        except: pass

    def run(self, cluster=None, verbose=False):
        """
        Runs all the tasks in a smart order with many at once. Will not return until all tasks are done.

        Giving a cluster (as a Cluster object) means that the cluster is used for any tasks that are added with can_run_on_cluster=True.
        
        Setting verbose to True will cause the time and the command to print whenever a new command is about to start.
        """

        # Checks
        if self.__conditional != None: raise ValueError('Tasks already running')
        if  len(self.inputs) == 0 and len(self.generators) == 0  and len(self.outputs) == 0 and len(self.cleanups) == 0 : return
        if (len(self.inputs) == 0 and len(self.generators) == 0) or (len(self.outputs) == 0 and len(self.cleanups) == 0): raise ValueError('Invalid set of tasks (likely cyclic)')
        prev_signal = None

        try:
            # Create basic variables and lock
            self._cluster = cluster
            self.__error = False
            self.__killing = False
            self.__conditional = Condition() # for locking access to Task.done, cpu_pressure, mem_pressure, next, last, log, and error
            self.__running = set()

            # Setup log
            done_tasks = self.__process_log() if exists(self.logname) else ()
            self.__log = open(self.logname, 'w', 0)
            for k,v in self.settings.iteritems(): self.__log.write("*"+k+"="+str(v)+"\n")
            # TODO: log overall inputs and outputs
            for dc in done_tasks:
                if verbose: print "Skipping " + dc[20:].strip()
                self.__log.write(dc+"\n")
            if verbose and len(done_tasks) > 0: print '-' * 80
            self._rusagelog = open(self.rusage_log, 'a', 1) if self.rusage_log else None

            # Calcualte the set of first and last tasks
            self.__calc_next() # These are the first tasks
            last = {self.outputs[f] for f in self.overall_outputs()}
            last |= self.cleanups
            if len(last) == 0: raise ValueError('Tasks are cyclic')
            self.__last = {t for t in last if not t.done}

            # Get the initial pressures
            self.__cpu_pressure = 0
            self.__mem_pressure = get_mem_used_by_tree() + 1*MB # assume that the creation of threads and everything will add some extra pressure

            # Set a signal handler
            try:
                from signal import signal, SIGUSR1
                prev_signal = signal(SIGUSR1, self.display_stats)
            except: pass

            # Keep running tasks in the tree until we have completed the root (which is self)
            with self.__conditional:
                while len(self.__last) != 0:
                    # Get next task (or wait until all tasks or finished or an error is generated)
                    while len(self.__last) > 0 and not self.__error:
                        task = self.__next_task()
                        if task != None: break
                        self.__conditional.wait(30) # wait until we have some available [without the timeout CTRL+C does not work and we cannot see if memory is freed up on the system]
                    if len(self.__last) == 0 or self.__error: break

                    # Run it
                    self.__running.add(task)
                    t = Thread(target=self.__run, args=(task,))
                    t.daemon = True
                    if verbose: print strftime(Tasks.__time_format) + " Running " + str(task)
                    t.start()
                    sleep(0) # make sure it starts

                # There was an error, let running tasks finish
                if self.__error and len(self.__running) > 0:
                    write_error("Waiting for other tasks to finish running.\nYou can terminate them by doing a CTRL+C.")
                    while len(self.__running) > 0:
                        self.__conditional.wait(60) # wait until a task stops [without the timeout CTRL+C does not work]

        except KeyboardInterrupt:

            # Terminate and kill tasks
            write_error("Terminating running tasks")
            with self.__conditional:
                self.__killing = True
                for t in self.__running:
                    try: t.terminate()
                    except: pass
                secs = 0
                while len(self.__running) > 0 and secs < 10:
                    self.__conditional.wait(1)
                    secs += 1
                for t in self.__running:
                    try: p.kill()
                    except: pass

        finally:
            # Cleanup
            if prev_signal: signal(SIGUSR1, prev_signal)
            if hasattr(self, '__log'):
                self.__log.close()
                del self.__log
            if hasattr(self, '_rusagelog'):
                if self._rusagelog: self._rusagelog.close()
                del self._rusagelog
            if hasattr(self, '__cpu_pressure'): del self.__cpu_pressure
            if hasattr(self, '__mem_pressure'): del self.__mem_pressure
            if hasattr(self, '__running'): del self.__running
            if hasattr(self, '__next'): del self.__next
            self.__conditional = None
            del self._cluster
            del self.__error
            del self.__killing

    def __process_log(self):
        """
        This looks at the previous log file and determines which commands do not need to be run this
        time through. This checks for changes in the commands themselves, when the commands were run
        relative to their output files, and more.
        """
        with open(self.logname, 'r+') as log: lines = [line.strip() for line in log]
        lines = [line for line in lines if len(line) != 0]
        #comments = [line for line in lines if line[0] == '#']
        # Note: this will take the last found setting/command with a given and silently drop the others
        settings = {s[0].strip():s[1].strip() for s in (line[1:].split('=',1) for line in lines if line[0] == '*')} # setting => value
        tasks = {line[20:].strip():line[:19] for line in lines if re.match('\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d\s', line)} # task string => date/time string
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
                if any((exists(f) and getmtime(f) >= datetime for f in t.inputs)) or any(not exists(f) for f in t.outputs):
                    changed.add(n)
        for n in changed.copy(): changed.update(str(t) for t in self.find(n).all_after()) # add every task that comes after a changed task

        # Mark as Done
        done_tasks = tasks.viewkeys() - changed
        for n in done_tasks: self.find(n).done = True
        return sorted((tasks[n] + " " + n) for n in done_tasks)
