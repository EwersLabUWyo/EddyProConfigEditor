from copy import copy
from collections.abc import Sequence
import datetime

from pandas import DataFrame

from eddyproconfigeditor import EddyproConfigEditor
from util import compare_configs

if __name__ == '__main__':
    base = EddyproConfigEditor('/Users/alex/Documents/Work/UWyo/Research/Flux Pipeline Project/Eddypro-ec-testing/investigate_eddypro/ini/base.eddypro')

    ref = base.copy()

    compare_configs(base, ref)

