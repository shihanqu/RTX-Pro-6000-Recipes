#!/usr/bin/env python3
"""Scan version-controlled/public candidate files, never ignored runtime output."""
import ipaddress
from pathlib import Path
import re
import subprocess

from config import ROOT

PATTERNS={
    'device_or_machine_uuid':r'\b(?:GPU-)?[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}\b',
    'account_home_path':r'/(?:home|Users)/[A-Za-z0-9_.-]+/',
    'access_token':r'\b(?:hf_|gh[pousr]_)[A-Za-z0-9]{20,}\b',
    'private_key':r'-----BEGIN (?:[A-Z]+ )?PRIVATE KEY-----',
    'hardware_mac':r'\b(?:[0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b',
}


def findings(text):
    result=[name for name,pattern in PATTERNS.items() if re.search(pattern,text)]
    for match in re.finditer(r'(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])',text):
        try:address=ipaddress.ip_address(match.group())
        except ValueError:continue
        if not address.is_loopback and address!=ipaddress.ip_address('0.0.0.0') and (
                address.is_private or address in ipaddress.ip_network((0x64400000,10))):
            result.append('private_network_address')
    return sorted(set(result))


def main():
    files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z',
        '--',ROOT.name,'README.md'],cwd=ROOT.parent).decode().split('\0')
    errors=[]
    for relative in files:
        if not relative:continue
        path=ROOT.parent/relative
        if not path.is_file():continue
        for item in findings(path.read_text(errors='replace')):errors.append((relative,item))
    if errors:raise SystemExit('Public privacy scan failed: '+repr(errors))
    print('Public-file privacy pattern scan passed. Also perform human review before publishing.')


if __name__=='__main__':main()
