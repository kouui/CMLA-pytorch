# -*- coding: utf-8 -*-
#-----------------------------------------------------------------------------
#
#-----------------------------------------------------------------------------

import datetime
from format_outcome import format_outcome

def print_separation_line(title=None):
    r"""

    print separation lines '---' with/without title

    """
    nBar = 40
    s = '-' * nBar
    if isinstance(title, str):
        title1 = ' ' + title + ' '
        i = 5
        assert len(title1) < (nBar-2*i)
        s = '\n' + s[:i] + title1 + s[i+len(title1):] + '\n'
    print(s)


def Loginfo(s_, fileObj_):

    timenow_ = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    ss_ = "[" + timenow_ + "]: " + s_ + "\n"
    fileObj_.write(ss_)
    fileObj_.flush()

def set_status(s_, status_, fileObj_=None):

    if status_ == "none":
        pass
    elif status_ == "terminal":
        print(s_)
    elif status_ == "log" and (fileObj_ is not None):
        Loginfo(s_, fileObj_=fileObj_)
    else:
        pass

def save_score_to_text(file, epoch, precision_as, recall_as, f1_as, precision_op, recall_op, f1_op):

    with open(file, 'a') as out:
        out.write(str(epoch))
        out.write('\n')
        out.write("aspect_precision: ")
        out.write(str(precision_as))
        out.write('\n')
        out.write("aspect_recall: ")
        out.write(str(recall_as))
        out.write('\n')
        out.write("aspect_f1: ")
        out.write(str(f1_as))
        out.write('\n')
        out.write("opinion_precision: ")
        out.write(str(precision_op))
        out.write('\n')
        out.write("opinion_recall: ")
        out.write(str(recall_op))
        out.write('\n')
        out.write("opinion_f1: ")
        out.write(str(f1_op))
        out.write('\n')

    format_outcome(inFile_=file, outFile_=file[:-4]+"_formated.txt")
