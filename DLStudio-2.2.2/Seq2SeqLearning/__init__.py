#!/usr/bin/env python

import sys

if sys.version_info[0] == 3:
    from DLStudio.DLStudio import __version__
    from DLStudio.DLStudio import __author__
    from DLStudio.DLStudio import __date__
    from DLStudio.DLStudio import __url__
    from DLStudio.DLStudio import __copyright__
    from Seq2SeqLearning.Seq2SeqLearning import Seq2SeqLearning
else:
    from DLStudio import __version__
    from DLStudio import __author__
    from DLStudio import __date__
    from DLStudio import __url__
    from DLStudio import __copyright__
    from Seq2SeqLearning import Seq2SeqLearning




