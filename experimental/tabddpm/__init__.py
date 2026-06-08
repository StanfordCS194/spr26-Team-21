"""TabDDPM synthesizer integration for Aperture.

A thin wrapper around Yandex Research's TabDDPM (https://github.com/yandex-research/tab-ddpm)
that loads a trained checkpoint and exposes the same interface as Aperture's bench
synthesizer base class. Training itself runs on a GPU box (we use Stanford Sherlock)
and is documented in README.md.

The actual checkpoint binaries live on HuggingFace, not in this repo. See MANIFEST.md
for the registry of published weights and which dataset each was trained on.
"""
