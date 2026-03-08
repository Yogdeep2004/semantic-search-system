#!/bin/bash
/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11 -m uvicorn api.main:app --host 0.0.0.0 --port 8000
