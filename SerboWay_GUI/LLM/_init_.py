from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

import numpy as np
import sounddevice as sd
from textual import events
from textual.app import App, ComposeResult
from textual.containers import Container
from textual.reactive import reactive
from textual.widgets import Button, RichLog, Static
from typing_extensions import override

from agents.voice import StreamedAudioInput, VoicePipeline