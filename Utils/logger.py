import logging
import coloredlogs

FORMAT = "%(asctime)s;%(name)s;%(levelname)s|%(message)s"
DATEF = "%H-%M-%S"
#logging.basicConfig(format=FORMAT, level=logging.INFO)
LEVEL_STYLES = dict(
    debug=dict(color="magenta"),
    info=dict(color="green"),
    verbose=dict(),
    warning=dict(color="blue"),
    error=dict(color="yellow"),
    critical=dict(color="red", bold=True),
)
LEVEL_STYLES_ALT = dict(
    debug=dict(color="magenta"),
    info=dict(color="cyan"),
    verbose=dict(),
    warning=dict(color="blue"),
    error=dict(color="yellow"),
    critical=dict(color="red", bold=True),
)

mylogger = logging.getLogger("Double_CUSUM_RL")
coloredlogs.install(level=logging.INFO,
                    fmt=FORMAT,
                    datefmt=DATEF,
                    level_styles=LEVEL_STYLES,
                    logger=mylogger)