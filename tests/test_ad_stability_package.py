"""Meaningful paired-statistics and integrity rejection checks."""
import hashlib
import json
import zipfile

import numpy as np
import pytest
from scripts.package_ad_v2_stability import statistics
from scripts.verify_ad_stability_zip import verify

def test_exact_signed_rank_keeps_losses_and_ties():
    percent,record=statistics([2,2,2,2,2],[1,1,1,1,1])
    assert record["exact_wilcoxon_p"]==.0625
    assert record["wins"]==5 and record["mean_paired_difference"]==1
    percent,record=statistics([2,2,2,2,2],[3,1,2,3,1])
    assert record["wins"]==2 and record["losses"]==2 and record["ties"]==1
    assert record["exact_wilcoxon_p"]==1
    assert record["mean_paired_improvement_percent"]==0

def test_integrity_rejects_tampered_bytes(tmp_path):
    path=tmp_path/"tampered.zip"
    with zipfile.ZipFile(path,"w") as archive:
        archive.writestr("raw/value.txt","changed")
        archive.writestr("checksums/SHA256.json",json.dumps({"raw/value.txt":hashlib.sha256(b"original").hexdigest()}))
    with pytest.raises(ValueError,match="Checksum mismatch"):verify(path)
