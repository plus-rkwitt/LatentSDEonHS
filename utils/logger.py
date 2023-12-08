"""File and console loggiong."""
import sys
import logging


class LogFormatter(logging.Formatter):
	COLOR_CODES = {
		logging.CRITICAL: "\033[1;35m", # bright/bold magenta
		logging.ERROR:    "\033[1;31m", # bright/bold red
		logging.WARNING:  "\033[1;33m", # bright/bold yellow
		logging.INFO:     "\033[0;37m", # white / light gray
		logging.DEBUG:    "\033[1;30m"  # bright/bold black / dark gray
	}

	RESET_CODE = "\033[0m"

	def __init__(self, color, *args, **kwargs):
		super(LogFormatter, self).__init__(*args, **kwargs)
		self.color = color

	def format(self, record, *args, **kwargs):
		if (self.color == True and record.levelno in self.COLOR_CODES):
			record.color_on  = self.COLOR_CODES[record.levelno]
			record.color_off = self.RESET_CODE
		else:
			record.color_on  = ""
			record.color_off = ""
		return super(LogFormatter, self).format(record, *args, **kwargs)


def set_up_logging(console_log_level, console_log_color, logfile_file, logfile_log_level, logfile_log_color, log_line_template):
	logger = logging.getLogger()
	logger.setLevel(logging.DEBUG)

	console_handler = logging.StreamHandler(sys.stdout)
	try:
		console_handler.setLevel(console_log_level.upper())
	except:
		print("Failed to set console log level: invalid level: '%s'" % console_log_level)
		return False

	console_formatter = LogFormatter(fmt=log_line_template, color=console_log_color)
	console_handler.setFormatter(console_formatter)
	logger.addHandler(console_handler)

	if logfile_file is not None:
		try:
			logfile_handler = logging.FileHandler(logfile_file, mode="w")
		except Exception as exception:
			print("Failed to set up log file: %s" % str(exception))
			return False

		try:
			logfile_handler.setLevel(logfile_log_level.upper())
		except:
			print("Failed to set log file log level: invalid level: '%s'" % logfile_log_level)
			return False

		logfile_formatter = LogFormatter(fmt=log_line_template, color=logfile_log_color)
		logfile_handler.setFormatter(logfile_formatter)
		logger.addHandler(logfile_handler)
	return True
