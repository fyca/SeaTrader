param(
  [Parameter(ValueFromRemainingArguments=$true)]
  [string[]]$Args
)
python "$PSScriptRoot/dashboard_ctl.py" @Args
