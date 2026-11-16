# Static PowerShell command parser (JSON Lines protocol on stdin/stdout).
#
# This process only *parses* text. It never runs, evaluates or interpolates the
# commands it is asked about: the text arrives as JSON data and is handed to
# System.Management.Automation.Language.Parser.ParseInput, which produces a pure
# AST.
#
# Request  (one JSON object per line): {"id": <int>, "command": "<shell text>"}
# Response (one JSON object per line):
#   {"id": <int>, "complete": <bool>, "dynamic": <bool>, "error": <string|null>,
#    "commands": [{"executable": "<name>", "wrappers": ["sudo"],
#                  "args": ["-Path", "out.txt"], "dynamic": <bool>}],
#    "redirections": [{"target": "out.txt", "append": <bool>}]}
#
# "complete" is only true when every operation that could run was extracted.
# "dynamic" reports that some value could not be known statically (a variable,
# an expandable string, a script block) even though the operation itself was
# identified.

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
    [Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
} catch {
    # Older hosts may refuse to reseat the console encodings; the protocol then
    # falls back to the inherited ones and the client decodes defensively.
}

$MaxNodes = 20000

function Get-StaticText($node) {
    if ($null -eq $node) { return $null }
    if ($node -is [System.Management.Automation.Language.StringConstantExpressionAst]) {
        return [string]$node.Value
    }
    if ($node -is [System.Management.Automation.Language.ConstantExpressionAst]) {
        return [string]$node.Value
    }
    return $null
}

function Test-NullTarget($node) {
    if ($node -is [System.Management.Automation.Language.VariableExpressionAst]) {
        return $node.VariablePath.UserPath.ToLowerInvariant() -eq 'null'
    }
    return $false
}

function Test-DirectiveCommand([string]$name) {
    # Commands that evaluate their argument as PowerShell source.
    return @('invoke-expression', 'iex', 'invoke-command', 'icm', 'start-job',
             'start-threadjob', 'enter-pssession', 'invoke-item') -contains $name
}

function Convert-Ast($text) {
    $tokens = $null
    $errors = $null
    $ast = [System.Management.Automation.Language.Parser]::ParseInput($text, [ref]$tokens, [ref]$errors)

    $state = @{ complete = $true; dynamic = $false; error = $null }

    if ($errors -and $errors.Count -gt 0) {
        $state.complete = $false
        $state.dynamic = $true
        $state.error = [string]$errors[0].Message
    }

    # The stop-parsing token ("--%") hands the rest of the line to the native
    # command verbatim. What the child process does with it is unknowable.
    foreach ($token in $tokens) {
        if ([string]$token.Text -eq '--%') {
            $state.complete = $false
            $state.dynamic = $true
            if (-not $state.error) { $state.error = 'stop-parsing token' }
        }
    }

    $commands = New-Object System.Collections.ArrayList
    $redirections = New-Object System.Collections.ArrayList

    $commandAsts = @($ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.CommandAst] }, $true))
    if ($commandAsts.Count -gt $MaxNodes) {
        return @{
            complete = $false; dynamic = $true; analysis_failed = $true
            error = 'command tree too large'
            commands = @(); redirections = @()
        }
    }

    foreach ($command in $commandAsts) {
        $elements = @($command.CommandElements)
        if ($elements.Count -eq 0) { continue }
        $head = Get-StaticText $elements[0]
        $commandDynamic = $false
        if ($null -eq $head -or $head.Length -eq 0) {
            # A variable, sub-expression or call operator in command position:
            # the executed program is not knowable statically.
            $state.complete = $false
            $state.dynamic = $true
            $commandDynamic = $true
            $head = [string]$elements[0].Extent.Text
        }
        $head = $head.Trim()
        if ($head.Length -eq 0) { continue }

        $args = New-Object System.Collections.ArrayList
        $literals = New-Object System.Collections.ArrayList
        for ($i = 1; $i -lt $elements.Count; $i++) {
            $element = $elements[$i]
            if ($element -is [System.Management.Automation.Language.CommandParameterAst]) {
                $parameter = '-' + [string]$element.ParameterName
                $value = Get-StaticText $element.Argument
                if ($null -ne $value) {
                    [void]$args.Add($parameter + ':' + $value)
                    [void]$literals.Add($true)
                } else {
                    if ($null -ne $element.Argument) {
                        $state.dynamic = $true
                        $commandDynamic = $true
                    }
                    [void]$args.Add($parameter)
                    [void]$literals.Add(($null -eq $element.Argument))
                }
                continue
            }
            $value = Get-StaticText $element
            if ($null -ne $value) {
                [void]$args.Add($value)
                [void]$literals.Add($true)
                continue
            }
            # Anything else (variable, expandable string, script block,
            # sub-expression, array/hashtable literal) contributes text we
            # cannot resolve before running.
            $state.dynamic = $true
            $commandDynamic = $true
            if ($element -is [System.Management.Automation.Language.ScriptBlockExpressionAst] -or
                $element -is [System.Management.Automation.Language.ScriptBlockAst]) {
                # A script block may run later, inside another command.
                $state.dynamic = $true
            }
            [void]$args.Add([string]$element.Extent.Text)
            [void]$literals.Add($false)
        }

        $lower = $head.ToLowerInvariant()
        if ((Test-DirectiveCommand $lower) -and ($args.Count -gt 0)) {
            $state.complete = $false
            $state.dynamic = $true
            if (-not $state.error) { $state.error = "$lower evaluates its argument as script" }
        }

        [void]$commands.Add([pscustomobject]@{
            executable = $head
            wrappers = @()
            args = @($args.ToArray())
            arg_literals = @($literals.ToArray())
            dynamic = $commandDynamic
        })

        foreach ($redirection in $command.Redirections) {
            if ($redirection -is [System.Management.Automation.Language.FileRedirectionAst]) {
                $literal = $true
                $location = Get-StaticText $redirection.Location
                if ($null -eq $location -and (Test-NullTarget $redirection.Location)) {
                    # `2>$null` discards output; it is not a file write.
                    [void]$redirections.Add([pscustomobject]@{
                        target = '$null'
                        append = [bool]$redirection.Append
                        literal = $true
                    })
                    continue
                }
                if ($null -eq $location) {
                    $state.complete = $false
                    $state.dynamic = $true
                    $literal = $false
                    $location = [string]$redirection.Location.Extent.Text
                }
                [void]$redirections.Add([pscustomobject]@{
                    target = [string]$location
                    append = [bool]$redirection.Append
                    literal = $literal
                })
            }
        }
    }

    # Function definitions hide calls we cannot see: the body is extracted as
    # potential commands, but nothing proves it is not invoked dynamically.
    if ($ast.Find({ param($n) $n -is [System.Management.Automation.Language.FunctionDefinitionAst] }, $true)) {
        $state.dynamic = $true
    }
    # Any variable reference means a value is resolved at run time. The
    # automatic constants ($null/$true/$false) are not variables in that sense.
    foreach ($variable in $ast.FindAll({ param($n) $n -is [System.Management.Automation.Language.VariableExpressionAst] }, $true)) {
        if (@('null', 'true', 'false') -notcontains $variable.VariablePath.UserPath.ToLowerInvariant()) {
            $state.dynamic = $true
            break
        }
    }

    return @{
        complete = $state.complete
        dynamic = $state.dynamic
        error = $state.error
        commands = @($commands.ToArray())
        redirections = @($redirections.ToArray())
    }
}

$reader = [Console]::In
while ($null -ne ($line = $reader.ReadLine())) {
    $line = $line.Trim()
    if ($line.Length -eq 0) { continue }
    $id = $null
    $response = $null
    try {
        $request = $line | ConvertFrom-Json
        $id = $request.id
        $text = [string]$request.command
        if ($text.Length -eq 0) {
            $response = @{ complete = $true; dynamic = $false; error = $null; commands = @(); redirections = @() }
        } else {
            $response = Convert-Ast $text
        }
    } catch {
        $response = @{
            complete = $false; dynamic = $true; analysis_failed = $true
            error = 'parser failure: ' + $_.Exception.Message
            commands = @(); redirections = @()
        }
    }
    $payload = @{
        id = $id
        complete = [bool]$response.complete
        analysis_failed = [bool]$response.analysis_failed
        dynamic = [bool]$response.dynamic
        error = $response.error
        commands = @($response.commands)
        redirections = @($response.redirections)
    }
    [Console]::Out.WriteLine(($payload | ConvertTo-Json -Compress -Depth 6))
    [Console]::Out.Flush()
}
