"""Launch the AWG control GUI and the spectrum analyzer GUI together,
SHARING one AWG connection.

Why sharing matters: the Proteus is reached over a raw TCP socket
(TCPIP::...::5025::SOCKET) and generally accepts only ONE session, so two
windows each calling AWG() would leave whichever connects second failing or
hanging. Here the AWG window owns the connection and the spectrum-analyzer
window borrows it through `awg_provider`.

What that buys you: real interactive control. Connect and drive the AWG by
hand in the AWG window -- set a frequency, change amplitude, run a sweep --
while watching the response live in the analyzer window, instead of only
being able to run the canned automated measurements. The analyzer window's
automated runs (chirp band scan / CW level check / mixer check) still work
and simply take over the shared AWG for their duration; they stop the output
when finished, so re-send from the AWG window to resume manual control.

Run:  python run_all.py
"""
import tkinter as tk

from gui import LabGUI
from specan_gui import SpecAnGUI

if __name__ == "__main__":
    root = tk.Tk()
    lab = LabGUI(root)

    sa_win = tk.Toplevel(root)
    # Late-bound on purpose: LabGUI.awg is None until its Connect button is
    # pressed, so hand over a callable rather than the (currently None) value.
    SpecAnGUI(sa_win, awg_provider=lambda: lab.awg)

    root.mainloop()
