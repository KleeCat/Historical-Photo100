$target = 'D:\HuaweiMoveData\Users\ihggk\Desktop\Historical-Photo100\.claude\minimize_issue'
$action = New-ScheduledTaskAction -Execute 'cmd.exe' -Argument "/c rmdir /s /q `"$target`""
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)
$settings = New-ScheduledTaskSettingsSet -ExecutionTimeLimit (New-TimeSpan -Minutes 1)
Register-ScheduledTask -TaskName 'DeleteMinimizeIssue' -Action $action -Trigger $trigger -Settings $settings -RunLevel Highest -Force | Out-Null
Start-Sleep 10
Unregister-ScheduledTask -TaskName 'DeleteMinimizeIssue' -Confirm:$false -ErrorAction SilentlyContinue
if (Test-Path $target) { Write-Output "FAIL: still exists" } else { Write-Output "Done" }
