"""

.. codeauthor:: Laurent Mertens <laurent.mertens@kuleuven.be>
"""
import os


class Config:
    #: Users system home dir
    ROOT_DIR = os.path.expanduser('/')
    HOME_DIR = os.path.expanduser('~')
    # HOME_DIR = "C:\\Laurent\\"
    DIR_BASE = os.path.join(HOME_DIR, "Work", "Projects", "AudioAnalysisItvs-JvdS")
    DIR_CODE = os.path.join(DIR_BASE, "Code")
    DIR_PROJECT = os.path.join(DIR_CODE, "SpeechTools")

    DIR_DATA = os.path.join(DIR_BASE, "Data")
    # Smallest Dutch file
    FILE_DEMO = os.path.join(DIR_DATA, "Dutch Recordings", "17_240701.mp3")

