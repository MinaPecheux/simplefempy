# Copyright 2019 Mina Pêcheux (mina.pecheux@gmail.com)
# ---------------------------
# Distributed under the MIT License:
# ==================================
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ==============================================================================
# [SimpleFEMPy] A basic Python PDE solver with the finite elements method
# ------------------------------------------------------------------------------
# Utils: Subpackage with various util tools
# ------------------------------------------------------------------------------
# logger.py - Logger manager (with a built-in static instance)
# ==============================================================================
import logging
import sys
import inspect
import copy

from simplefempy.settings import LIB_SETTINGS, TK_INSTANCE

class SimpleFemPyError(Exception):
    pass

# ---- ** ----
# From: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output
K, R, G, Y, B, M, C, W = range(8)

# These are the sequences need to get colored ouput
RESET_SEQ = '\033[0m'
COLOR_SEQ = '\033[0;%dm'
BOLD_SEQ  = '\033[1m'

def formatter_message(message, use_color=True):
    if use_color:
        message = message.replace('$RESET', RESET_SEQ).replace('$BOLD', BOLD_SEQ)
    else:
        message = message.replace('$RESET', '').replace('$BOLD', '')
    return message

COLORS = { 'WARNING': Y, 'INFO': C, 'DEBUG': M, 'ERROR': R }

class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color=True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        rec = copy.copy(record) # copy the record object to avoid confusion between streams
        lvl = rec.levelname
        if self.use_color and lvl in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[lvl]) + lvl + RESET_SEQ
            rec.levelname = levelname_color
        return logging.Formatter.format(self, rec)
# ---- ** ----

class Logger(object):
    """Object to log information according to given level (uses Python
    base logger).

    Parameters
    ----------
    logging_level : str or int, optional
        Level of display (to be checked).
    prefix : str, optional
        Specific project name to show at the beginning of all logs. 'mplibs' by
        default.
    use_color : bool, optional
        If true, color is used to distinguish between the different levels of
        logging.
    stream : IO.stream or None, optional
        If it is not None, then this Logger's output is redirected to the given
        stream. Else, logging is done in the sys.stdout stream.
    abort_errors : bool, optional
        If false (by default), critical errors trigger an instant exit of the
        program.
    """

    LOGGING_LEVELS = {
        'error': 40, 'warning': 30, 'info': 20, 'debug': 10,
        40: 'error', 30: 'warning', 20: 'info', 10: 'debug'
    }

    def __init__(self, logging_level='info', prefix='FEM', use_color=True,
                 stream=None, abort_errors=False):
        self.prefix = prefix
        self.logger = logging.getLogger()
        self.set_level(logging_level)
        self.set_stream(stream, use_color)

        self.abort_errors = abort_errors

    @staticmethod
    def _check_logging_level(logging_level, no_switch=False):
        """Returns the logging level after checking if it is valid or switching to
        default 'info' mode.

        Parameters
        ----------
        logging_level : str or int
            Level of display. If it is a string, it is converted to the matching
            values if it is one of: 'critical', 'error', 'warning', 'info', 'debug';
            else if switching is enabled, the default setting 'info' is taken. If it
            is an int, it must be one of: 50, 40, 30, 20, 10; else if switching is
            enabled the default setting 20 is taken.
        no_switch : bool, optional
            If true, an unknown logging level will be returned as is (cannot be used
            but can be spotted as wrong). Else, it is switched to the default setting
            ('info' mode).
        """
        if no_switch:
            if isinstance(logging_level, str):
                try: logging_level = Logger.LOGGING_LEVELS[logging_level]
                except KeyError: pass
            return logging_level
        else:
            if isinstance(logging_level, int) and not logging_level in [10,20,30,40,50]:
                logging_level = 20
            if isinstance(logging_level, str):
                try: logging_level = Logger.LOGGING_LEVELS[logging_level]
                except KeyError: logging_level = Logger.LOGGING_LEVELS['info']
            return logging_level

    def get_level(self):
        """Gets the current display level."""
        return self.logger.getEffectiveLevel()

    def set_level(self, logging_level):
        """Sets the display level.

        Parameters
        ----------
        logging_level : str or int, optional
            Level of display (to be checked).
        """
        l = Logger._check_logging_level(logging_level)
        self.logger.setLevel(l)
        
    def set_stream(self, stream, use_color):
        """Sets the output stream.
        
        Parameters
        ----------
        stream : IO.stream
            Stream to output to.
        """
        indent = 18 if use_color else 7
        form = '[$BOLD{}$RESET.%(levelname){}s] %(message)s'.format(self.prefix, indent)
        color_formatter = ColoredFormatter(formatter_message(form, use_color),
                                           use_color)
        stream_handler  = logging.StreamHandler(stream)
        stream_handler.setFormatter(color_formatter)

        l = Logger._check_logging_level(self.get_level())
        stream_handler.setLevel(l)

        self.logger.addHandler(stream_handler)
        
    def set_errors(self, abort):
        """Sets the 'abort_errors' flag (if false, errors trigger an exit of the
        program; else, the error must be handled elsewhere).
        
        Parameters
        ----------
        abort : bool
            New value for the 'abort_errors' flag.
        """
        self.abort_errors = abort

    def log(self, msg, level='info', stackoffset=2):
        """Logs a message.

        Parameters
        ----------
        msg : str
            Message to display.
        level : str or int, optional
            Level of logging for the message (can be: error', 'warning', 'info'
            or 'debug' for a string; or 40, 30, 20 or 10 for an int).
        """
        try: stackdata = inspect.stack()[1+stackoffset]
        except IndexError: stackdata = inspect.stack()[-1]
        # use tuple direct indexing for Python 2.x compatibility
        caller_file = stackdata[1].split('/')[-1]
        caller_func = stackdata[3]
        lineno      = stackdata[2]
        if caller_func == '<module>': caller_func = ''
        else: caller_func = ' - {}()'.format(caller_func)
        msg = '({}:{}{}): {}'.format(caller_file, lineno, caller_func, msg)
        l = Logger._check_logging_level(level, no_switch=True)
        if l == logging.DEBUG: self.logger.debug(msg)
        elif l == logging.INFO: self.logger.info(msg)
        elif l == logging.WARNING: self.logger.warning(msg)
        elif l == logging.ERROR: self.logger.error(msg)
        else: self.error('Unknown level of logging: "%s".' % level)

    def error(self, msg, stackoffset=2):
        """Warns the user of a fatal error and exists the program with error
        return code (1).

        Parameters
        ----------
        msg : str
            Error message to display.
        """
        self.log(msg, level='error', stackoffset=stackoffset+1)
        if TK_INSTANCE['app'] is not None: TK_INSTANCE['app'].exit()
        if not self.abort_errors:
            sys.exit(1)
        else:
            raise SimpleFemPyError

    @staticmethod
    def sget_level():
        """Static equivalent of :func:`.Logger.get_level`."""
        return STATIC_LOGGER.get_level()

    @staticmethod
    def sset_level(logging_level):
        """Static equivalent of :func:`.Logger.set_level`."""
        STATIC_LOGGER.set_level(logging_level)

    @staticmethod
    def sset_stream(stream, use_color=False):
        """Static equivalent of :func:`.Logger.set_stream`."""
        STATIC_LOGGER.set_stream(stream, use_color)

    @staticmethod
    def sset_errors(abort):
        """Static equivalent of :func:`.Logger.set_errors`."""
        STATIC_LOGGER.set_errors(abort)

    @staticmethod
    def slog(msg, level='info', stackoffset=1):
        """Static equivalent of :func:`.Logger.log`."""
        STATIC_LOGGER.log(msg, level, stackoffset=stackoffset+1)

    @staticmethod
    def serror(msg, stackoffset=1):
        """Static equivalent of :func:`.Logger.error`."""
        STATIC_LOGGER.error(msg, stackoffset=stackoffset+1)

    @staticmethod
    def sset_prefix(prefix):
        """Sets the prefix of the static logger.

        Parameters
        ----------
        prefix : str
            Specific project name to show at the beginning of all logs.
        """
        STATIC_LOGGER.prefix = prefix
        formatter = logging.Formatter('[ {} ] . %(asctime)s :: %(levelname)s '
                                      ':: %(message)s'.format(prefix),
                                      datefmt='%Y-%m-%d %H:%M:%S')
        STATIC_LOGGER.stream_handler.setFormatter(formatter)

STATIC_LOGGER = Logger()
