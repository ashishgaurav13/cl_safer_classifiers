import sys, datetime, os
import time

# https://stackoverflow.com/questions/17866724/
# python-logging-print-statements-while-having-them-print-to-stdout
# Print something while also logging to another file.
class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        for f in self.files:
            f.flush()

# Use __main__.__file__, timestamp, seed to create a .txt
# log file, such that when something is printed, it shows up
# on stdout as well in that .txt log file; If overwrite_name
# is given, then we replace __main__.__file__ with that
def setup_logging(seed, overwrite_name = None):
    if not os.path.exists("logs"):
        os.mkdir("logs")
        print("os.mkdir: logs")
    import __main__
    logfile = __main__.__file__
    logfile = logfile.rstrip(".py")
    if overwrite_name:
        assert(type(overwrite_name) == str)
        logfile = overwrite_name
    print('Detected seed: %d' % seed)
    logfile = logfile + datetime.datetime.now().strftime("_%Y_%m_%d_%H_%M_%S") + "_seed%d.txt" % seed
    logfile = open(os.path.join("logs", logfile), "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, logfile)

# Gives an initial time using time.time()
def timer():
    print("Timer started!")
    return time.time()

# Prints elapsed time in human-readable units, given a start
# datetime object
def timer_done(start_time):
    print("Elapsed: %s" % str(datetime.timedelta(seconds = int(time.time()-start_time))))

def const_str(x):
    if x <= 1000: return "%g" % x
    else: return "%d" % int(x)