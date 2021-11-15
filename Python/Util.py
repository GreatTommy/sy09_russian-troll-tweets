import sys
import time
from math import ceil

"""
    ////////////////////////////////////// PROGRESS ////////////////////////////////
"""

def print_progress_percent(current, begin=0, end=100):
    percent=((current/(end-begin)) *100)+1
    sys.stdout.write("\rProgress-------- %d%%" % percent)
    sys.stdout.flush()


class Progress:
    _begin=None
    _end=None
    _range=None

    _name=None

    _start_time=None

    _i=0

    def __init__(self, begin=0, end=100, name=None, i=0):
        self._begin=begin
        self._end=end
        self._range=self._end-self._begin
        self._name=name
        self._i=i
        self._opt_percent=0

        if self._name:
            print(f" ", flush=True)
            sys.stdout.write(f"nb_it={self._end-self._begin}\n\rProgress({self._name})---- 0 % ---- 0 min ---- encore . min")
        else:
            print(f" ", flush=True)
            sys.stdout.write(f"nb_it={self._end-self._begin}\n\rProgress---- 0 % ---- 0 min ---- encore . min")
        sys.stdout.flush()

    def __del__(self):
        print("\n", flush=True)


    _opt_percent=0
    def print_progress_percent(self, current):
        if current-self._begin==0:
            if self._name:
                sys.stdout.write(f"\rProgress({self._name})---- 0 % ---- 0 min ---- encore . min")
            else:
                sys.stdout.write(f"\rProgress---- 0 % ---- 0 min ---- encore . min")
            sys.stdout.flush()
            return

        percent=ceil(((current-self._begin)/self._range) *100)

        if self._opt_percent==int(percent):
            return
        else:
            self._opt_percent=int(percent)

        if self._start_time==None:
            self._start_time=time.time()
        temps=time.time() - self._start_time

        if self._name:
            sys.stdout.write(f"\rProgress({self._name})---- {int(percent)} % ---- {int(temps/60)} min ---- encore {int(((self._end-current)/(current-self._begin))*temps/60)} min")
        else:
            sys.stdout.write(f"\rProgress---- {int(percent)} % ---- {int(temps/60)} min ---- encore {int(((self._end-current)/(current-self._begin))*temps/60)} min")
        sys.stdout.flush()

    def print_progress_percent_ghost(self, current, var):
        if current-self._begin==0:
            if self._name:
                sys.stdout.write(f"\rProgress({self._name})---- 0 % ---- 0 min ---- encore . min")
            else:
                sys.stdout.write(f"\rProgress---- 0 % ---- 0 min ---- encore . min")
            sys.stdout.flush()
            return var

        percent=ceil(((current-self._begin)/self._range) *100)
        if self._opt_percent==int(percent):
            return var
        else:
            self._opt_percent=int(percent)

        if self._start_time==None:
            self._start_time=time.time()
        temps=time.time() - self._start_time

        if self._name:
            sys.stdout.write(f"\rProgress({self._name})---- {int(percent)} % ---- {int(temps/60)} min ---- encore {int(((self._end-current)/(current-self._begin))*temps/60)} min")
        else:
            sys.stdout.write(f"\rProgress---- {int(percent)} % ---- {int(temps/60)} min ---- encore {int(((self._end-current)/(current-self._begin))*temps/60)} min")
        sys.stdout.flush()
        return var

    def init_incremental_progress_ghost(self, i=0):
        if self._name:
            sys.stdout.write(f"\rProgress({self._name})---- 0 % ---- 0 min ---- encore . min")
        else:
            sys.stdout.write(f"\rProgress---- 0 % ---- 0 min ---- encore . min")
        self._i=i
        self._opt_percent=0

    def print_incremental_progress_percent(self):
        self._i+=1
        current=self._i
        if current-self._begin==0:
            if self._name:
                sys.stdout.write(f"\rProgress({self._name})---- 0 % ---- 0 min ---- encore . min")
            else:
                sys.stdout.write(f"\rProgress---- 0 % ---- 0 min ---- encore . min")
            sys.stdout.flush()
            return

        percent=ceil(((current-self._begin)/self._range) *100)
        if self._opt_percent==int(percent):
            return
        else:
            self._opt_percent=int(percent)

        if self._start_time==None:
            self._start_time=time.time()
        temps=time.time() - self._start_time

        if self._name:
            sys.stdout.write(f"\rProgress({self._name})---- {int(percent)} % ---- {int(temps/60)} min ---- encore {int(((self._end-current)/(current-self._begin))*temps/60)} min")
        else:
            sys.stdout.write(f"\rProgress---- {int(percent)} % ---- {int(temps/60)} min ---- encore {int(((self._end-current)/(current-self._begin))*temps/60)} min")
        sys.stdout.flush()
        return

    def print_incremental_progress_percent_ghost(self, var):
        self._i+=1
        current=self._i
        if current-self._begin==0:
            if self._name:
                sys.stdout.write(f"\rProgress({self._name})---- 0 % ---- 0 min ---- encore . min")
            else:
                sys.stdout.write(f"\rProgress---- 0 % ---- 0 min ---- encore . min")
            sys.stdout.flush()
            return var

        percent=ceil(((current-self._begin)/self._range) *100)
        if self._opt_percent==int(percent):
            return var
        else:
            self._opt_percent=int(percent)

        if self._start_time==None:
            self._start_time=time.time()
        temps=time.time() - self._start_time

        if self._name:
            sys.stdout.write(f"\rProgress({self._name})---- {int(percent)} % ---- {int(temps/60)} min ---- encore {int(((self._end-current)/(current-self._begin))*temps/60)} min")
        else:
            sys.stdout.write(f"\rProgress---- {int(percent)} % ---- {int(temps/60)} min ---- encore {int(((self._end-current)/(current-self._begin))*temps/60)} min")
        sys.stdout.flush()
        return var






"""
    ////////////////////////////////////// OTHER ////////////////////////////////
"""

def add_labels(x, y, labels, ax=None):
    """
    Ajoute les étiquettes `labels` aux endroits définis par `x` et `y`.
    """
    if ax is None:
        ax = plt.gca()
    for x, y, label in zip(x, y, labels):
        ax.annotate(
            label, [x, y], xytext=(10, -5), textcoords="offset points",
        )
    return ax


def is_df(obj):
    return type(obj)==pd.DataFrame













"""
    ////////////////////////////////////// MAIN ////////////////////////////////
"""

def __main__():
    progress=Progress(name="test")
    for i in range(100):
        progress.print_incremental_progress_percent_ghost(i)


















