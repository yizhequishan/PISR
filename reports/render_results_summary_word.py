"""Use the packaged DOCX rasterizer with Word as Windows PDF backend.

The managed Windows runtime has no bundled LibreOffice. This adapter uses a
separate invisible Word instance to export only the specified report, then
delegates PDF rasterization and page naming to the packaged render_docx.py.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

RUNTIME=Path(r'C:\Users\yizhe\.cache\codex-runtimes\codex-primary-runtime')
RENDER=Path(r'C:\Users\yizhe\.codex\plugins\cache\openai-primary-runtime\documents\26.909.61513\skills\documents\render_docx.py')
POPPLER=RUNTIME/'dependencies/native/poppler/Library/bin'
os.environ['PATH']=str(POPPLER)+os.pathsep+os.environ.get('PATH','')
spec=importlib.util.spec_from_file_location('packaged_render_docx',RENDER)
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)

def word_pdf(doc_path,user_profile,convert_tmp_dir,stem,verbose):
    pdf=Path(convert_tmp_dir)/(stem+'.pdf')
    # Late-bound IDispatch avoids the machine's incompatible Office .NET PIA.
    script=Path(convert_tmp_dir)/'export_word.vbs'
    script.write_text('''Option Explicit
Dim app, doc, failure
On Error Resume Next
Set app = CreateObject("Word.Application")
If Err.Number <> 0 Then
  WScript.Echo "CREATE ERROR " & Err.Description
  WScript.Quit 1
End If
app.Visible = False
app.DisplayAlerts = 0
app.AutomationSecurity = 3
Err.Clear
Set doc = app.Documents.Open(WScript.Arguments(0), False, True, False)
If Err.Number <> 0 Then
  failure = Err.Description
  app.Quit 0
  WScript.Echo "OPEN ERROR " & failure
  WScript.Quit 1
End If
doc.Repaginate
Err.Clear
doc.ExportAsFixedFormat WScript.Arguments(1), 17
If Err.Number <> 0 Then
  failure = Err.Description
  doc.Close 0
  app.Quit 0
  WScript.Echo "EXPORT ERROR " & failure
  WScript.Quit 1
End If
WScript.Echo "WORD_PAGES=" & doc.ComputeStatistics(2)
doc.Close 0
app.Quit 0
Set doc = Nothing
Set app = Nothing
''',encoding='utf-16')
    p=subprocess.run(['cscript.exe','//Nologo',str(script),str(Path(doc_path).resolve()),str(pdf)],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
    log=p.stdout+'\n'+p.stderr
    if verbose:print(log)
    return (str(pdf) if p.returncode==0 and pdf.exists() else '',log)

renderer.convert_to_pdf=word_pdf
if __name__=='__main__':renderer.main()
