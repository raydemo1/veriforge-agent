
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from harness_code_agent.agent.context import count_text_tokens, count_tokens
from harness_code_agent.agent.observations import FactTracker, ObservationStore
from harness_code_agent.runtime.builtins.filesystem import read_file
from harness_code_agent.runtime.tool_result import ToolResult
from harness_code_agent.workspace.service import WorkspaceService


def observe(root, output, *, result=None):
    store = ObservationStore(root / '.harness' / 'observations' / 'session')
    result = result or ToolResult(tool='test', status='success', output=output)
    observation = store.create(tool=result.tool, args={}, result=result, fact_tracker=FactTracker())
    return observation, store.observed_message(observation, result)


def test_small_output_is_inline_without_artifact(tmp_path):
    observation, message = observe(tmp_path, 'short result')
    assert 'short result' in message
    assert observation.raw_output_path is None
    assert not list(tmp_path.rglob('*.txt'))


def test_large_output_has_reference_and_bounded_head_tail_preview(tmp_path):
    output = 'HEAD_SENTINEL\n' + 'the quick brown fox jumps ' * 3000 + 'MIDDLE_SENTINEL' + 'the quick brown fox jumps ' * 3000 + '\nTAIL_SENTINEL'
    observation, message = observe(tmp_path, output)
    assert 'HEAD_SENTINEL' in message
    assert 'TAIL_SENTINEL' in message
    assert 'MIDDLE_SENTINEL' not in message
    head = message.split('--- output head ---\n', 1)[1].split('\n\n[middle omitted;', 1)[0]
    tail = message.split('--- output tail ---\n', 1)[1]
    assert count_text_tokens(head) <= 500
    assert count_text_tokens(tail) <= 500
    assert 'read_file' in message
    assert str(tmp_path) not in message
    assert count_tokens([{'role': 'tool', 'content': message}]) <= 5000
    assert observation.raw_output_path.read_text(encoding='utf-8') == output


def test_reopening_store_does_not_overwrite_output(tmp_path):
    first, _ = observe(tmp_path, 'first ' * 20000)
    second, _ = observe(tmp_path, 'second ' * 20000)
    assert first.raw_output_path != second.raw_output_path
    assert first.raw_output_path.read_text(encoding='utf-8') == 'first ' * 20000


def test_observation_index_eviction_keeps_artifacts(tmp_path):
    store = ObservationStore(tmp_path / 'outputs')
    first = store.create(tool='test', args={}, result=ToolResult(tool='test', status='success', output='large ' * 20000), fact_tracker=FactTracker())
    for _ in range(300):
        store.create(tool='test', args={}, result=ToolResult(tool='test', status='success', output='small'), fact_tracker=FactTracker())
    assert first.raw_output_path.exists()
    assert len(store.observations) == 300


def test_file_pages_reconstruct_long_unicode_line(tmp_path):
    source = ('中文🙂 the quick brown fox ' * 4000) + '\nlast line'
    (tmp_path / 'source.txt').write_text(source, encoding='utf-8')
    context = SimpleNamespace(workspace=WorkspaceService(root=tmp_path))
    arguments = {'path': 'source.txt', 'start_line': 1, 'max_lines': 2}
    pages = []
    for _ in range(100):
        result = read_file(**arguments, tool_context=context)
        assert result.status == 'success'
        observation, message = observe(tmp_path, '', result=result)
        assert observation.raw_output_path is None
        assert count_tokens([{'role': 'tool', 'content': message}]) <= 5000
        pages.append(result.output)
        next_read = result.metadata.get('next_read')
        if not next_read:
            break
        assert next_read != arguments
        arguments = next_read
    else:
        raise AssertionError('pagination did not finish')
    assert ''.join(pages) == source


def test_file_pages_continue_across_line_windows_and_keep_final_newlines(tmp_path):
    source = ''.join(f'row {index}\n' for index in range(4100)) + '\n\n'
    (tmp_path / 'rows.txt').write_text(source, encoding='utf-8')
    context = SimpleNamespace(workspace=WorkspaceService(root=tmp_path))
    arguments = {'path': 'rows.txt', 'max_lines': 2000}
    pages = []
    for _ in range(100):
        result = read_file(**arguments, tool_context=context)
        assert result.status == 'success'
        pages.append(result.output)
        if not result.metadata.get('next_read'):
            break
        assert result.metadata['next_read'] != arguments
        arguments = result.metadata['next_read']
    else:
        raise AssertionError('pagination did not finish')
    assert ''.join(pages) == source


def test_streamed_small_output_is_inlined_and_temporary_file_removed(tmp_path):
    directory = tmp_path / '.harness' / 'observations' / 'session'
    directory.mkdir(parents=True)
    artifact = directory / 'shell_output_small.txt'
    artifact.write_text('small captured result', encoding='utf-8')
    result = ToolResult(tool='run_bash', status='success', output='preview', return_code=0,
                        metadata={'artifact_path': str(artifact), 'output_spilled': True})
    observation, message = observe(tmp_path, '', result=result)
    assert 'small captured result' in message
    assert observation.raw_output_path is None
    assert not artifact.exists()


def test_structured_failure_fields_remain_visible(tmp_path):
    result = ToolResult(tool='test', status='failed', output='partial', error='important error', return_code=92371)
    _, message = observe(tmp_path, '', result=result)
    assert 'status: failed' in message
    assert 'important error' in message
    assert '92371' in message


def test_long_arguments_are_not_repeated_in_result(tmp_path):
    store = ObservationStore(tmp_path / 'outputs')
    result = ToolResult(tool='run_bash', status='success', output='ok', return_code=0)
    observation = store.create(tool='run_bash', args={'command': 'COMMAND_SECRET' * 10000},
                               result=result, fact_tracker=FactTracker())
    assert 'COMMAND_SECRET' not in observation.message
    assert observation.raw_output_path is None


def test_mcp_text_and_resource_tail_survive_budgeting(tmp_path):
    from harness_code_agent.runtime.mcp import (
        McpToolBinding,
        _mcporter_json_to_tool_result,
        mcp_result_to_tool_result,
    )

    text = 'the quick brown fox jumps ' * 4000 + 'MCP_TAIL_SENTINEL'
    incoming = SimpleNamespace(content=[SimpleNamespace(type='text', text=text)], structuredContent=None, isError=False)
    result = mcp_result_to_tool_result('mcp_test', incoming)
    observation, message = observe(tmp_path, '', result=result)
    assert 'MCP_TAIL_SENTINEL' in message
    assert 'MCP_TAIL_SENTINEL' in observation.raw_output_path.read_text(encoding='utf-8')
    binding = McpToolBinding('mcp_test', 'server', 'tool', 'description', {}, 'read')
    resource = _mcporter_json_to_tool_result(binding, {'content': [{'type': 'resource', 'resource': {
        'uri': 'file:///example', 'mimeType': 'text/plain', 'text': text}}]})
    assert text in resource.output


def test_web_and_skill_text_are_not_cut_before_budgeting(tmp_path):
    from harness_code_agent.runtime.builtins.filesystem import read_skill_file
    from harness_code_agent.runtime.builtins.web import web_fetch

    text = 'the quick brown fox jumps ' * 4000 + 'DOCUMENT_TAIL_SENTINEL'
    with patch('urllib.request.urlopen', return_value=SimpleNamespace(read=lambda: ('<p>' + text + '</p>').encode())):
        result = web_fetch('https://example.test')
    observation, message = observe(tmp_path, '', result=result)
    assert 'DOCUMENT_TAIL_SENTINEL' in message
    assert 'DOCUMENT_TAIL_SENTINEL' in observation.raw_output_path.read_text(encoding='utf-8')
    with patch('pathlib.Path.exists', return_value=True), patch('pathlib.Path.is_file', return_value=True), patch('pathlib.Path.read_text', return_value=text):
        result = read_skill_file('catalog/test/SKILL.md')
    assert result.output == text


@pytest.mark.parametrize('stdout,stderr', [('', 'ERROR_STDERR_ONLY'), ('stdout', ' ' * 2000 + 'ERROR_MIDDLE' + ' ' * 2000)])
def test_streamed_stderr_is_preserved(tmp_path, stdout, stderr):
    from harness_code_agent.workspace.shell_session import (
        _BoundedPipeSink,
        _stream_sinks_to_artifact,
        combine_output,
    )

    out = _BoundedPipeSink(preview_chars=256, spool_dir=tmp_path)
    err = _BoundedPipeSink(preview_chars=256, spool_dir=tmp_path)
    try:
        out.feed(stdout)
        err.feed(stderr)
        out.finish_writing()
        err.finish_writing()
        path, preview, *_ = _stream_sinks_to_artifact(out, err, artifact_dir=tmp_path)
        assert Path(path).read_text(encoding='utf-8') == combine_output(stdout, stderr)
        if not stdout:
            assert 'ERROR_STDERR_ONLY' in preview
    finally:
        out.discard()
        err.discard()


def test_persistent_capture_artifact_matches_inline_layout(tmp_path):
    from harness_code_agent.workspace.shell_session import (
        _stream_capture_to_artifact,
        combine_output,
    )

    stdout = '  中文🙂 line\n' * 7000
    stderr = '\nerror 中文🙂\n  '
    raw = tmp_path / 'capture.raw'
    raw.write_bytes(('OUT\r\n' + stdout + '\nERR\r\n' + stderr + '\nEXIT:7\r\n').encode('utf-8'))
    built = _stream_capture_to_artifact(raw, artifact_dir=tmp_path / 'artifacts',
                                      stdout_marker=b'OUT', stderr_marker=b'ERR', exit_marker=b'EXIT',
                                      preview_bytes=2000)
    expected = combine_output(stdout, stderr)
    assert Path(built.path).read_bytes() == expected.encode('utf-8')
    assert built.exit_code == 7
    assert built.total_chars == len(expected)
    assert built.total_bytes == len(expected.encode('utf-8'))


def test_background_log_spills_only_when_long_and_keeps_its_beginning(tmp_path):
    from harness_code_agent.workspace.shell_jobs import ShellOutputBuffer

    buffer = ShellOutputBuffer(tmp_path)
    buffer.append('START_MARKER\n')
    assert buffer.path is None
    buffer.append('the quick brown fox jumps ' * 10000)
    buffer.append('END_MARKER')
    assert buffer.path.is_file()
    assert buffer.read(0, 12) == 'START_MARKER'
    assert buffer.read_tail(10) == 'END_MARKER'
    assert 'START_MARKER' in buffer.path.read_text(encoding='utf-8')


def test_timeout_wrapper_keeps_captured_output(tmp_path):
    from harness_code_agent.runtime.builtins.shell import run_bash
    from harness_code_agent.workspace.shell_session import ShellResult

    shell = SimpleNamespace(run=lambda *a, **kw: ShellResult(stdout='PARTIAL_STDOUT', stderr='PARTIAL_STDERR', exit_code=130, timed_out=True), close=lambda: None)
    with patch('harness_code_agent.workspace.shell_session.PersistentShellSession', return_value=shell), patch('harness_code_agent.runtime.builtins.shell._requires_one_shot_powershell', return_value=False):
        result = run_bash('echo hello', timeout=1, tool_context=SimpleNamespace(workspace=SimpleNamespace(root=tmp_path)))
    assert result.status == 'failed'
    assert 'PARTIAL_STDOUT' in result.output
    assert 'PARTIAL_STDERR' in result.output


def test_artifact_copy_is_verified_and_cannot_escape_workspace(tmp_path):
    from harness_code_agent.runtime.output_artifacts import copy_artifact, file_digest

    source, target = tmp_path / 'source', tmp_path / 'target'
    artifact = source / '.harness/observations/session/main/output.txt'
    artifact.parent.mkdir(parents=True)
    artifact.write_text('original', encoding='utf-8')
    reference = {'path': artifact.relative_to(source).as_posix(), 'sha256': file_digest(artifact)}
    copy_artifact(source, target, reference)
    assert (target / reference['path']).read_text(encoding='utf-8') == 'original'
    artifact.write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='damaged'):
        copy_artifact(source, target, reference)
    with pytest.raises(ValueError):
        copy_artifact(source, target, {'path': '../outside.txt'})


def test_special_token_strings_are_plain_tool_content(tmp_path):
    _, message = observe(tmp_path, '<|endoftext|>')
    assert '<|endoftext|>' in message


def test_final_message_limit_includes_headers(tmp_path):
    for repeats, expected_file in [(4800, False), (5100, True)]:
        observation, message = observe(tmp_path, ' a' * repeats)
        assert (observation.raw_output_path is not None) == expected_file
        assert count_tokens([{'role': 'tool', 'content': message}]) <= 5000


def make_context(root, session_id):
    from harness_code_agent.runtime.builtins.registry import BUILTIN_TOOL_REGISTRY
    from harness_code_agent.runtime.permissions import PermissionPolicy
    from harness_code_agent.runtime.tool_context import ToolContext
    from harness_code_agent.sessions.events import EventBus

    return ToolContext(workspace=WorkspaceService(root=root), permission_policy=PermissionPolicy(mode='danger-full-access'),
                       event_bus=EventBus(), session_id=session_id, tool_registry=BUILTIN_TOOL_REGISTRY.copy())


def make_conversation(context):
    from harness_code_agent.agent.conversation import Agent

    with patch('harness_code_agent.agent.conversation.get_client', return_value=SimpleNamespace(close=lambda: None)):
        return Agent('test', 'system', tool_context=context).start_conversation()


def test_native_powershell_multiline_artifact_preserves_unicode(tmp_path):
    import os
    import sys

    from harness_code_agent.runtime.builtins.shell import run_bash
    from harness_code_agent.workspace.shell_session import (
        windows_shell_kind,
        windows_shell_path,
    )

    if os.name != 'nt' or windows_shell_kind() != 'pwsh' or not windows_shell_path():
        pytest.skip('requires Windows PowerShell 7')
    source = 'HEAD_NATIVE\n' + '中文🙂ABCDE\n' * 4000 + 'TAIL_NATIVE'
    (tmp_path / 'emit.py').write_text('import sys\nsys.stdout.write(' + repr(source) + ')', encoding='utf-8')
    context = make_context(tmp_path, 'native_session')
    conversation = make_conversation(context)
    try:
        result = run_bash("& '" + sys.executable.replace("'", "''") + "' -X utf8 'emit.py'",
                          runtime_state=conversation.runtime_state, tool_context=context)
        assert result.status == 'success' and result.return_code == 0
        observation = conversation.record_tool_result('native', result)
        assert observation.raw_output_path.read_text(encoding='utf-8') == source
        assert 'HEAD_NATIVE' in observation.message and 'TAIL_NATIVE' in observation.message
        assert count_tokens([{'role': 'tool', 'content': observation.message}]) <= 5000
        arguments = {'path': observation.artifact_reference['path']}
        pages = []
        while True:
            page = read_file(**arguments, tool_context=context)
            assert page.status == 'success'
            pages.append(page.output)
            if not page.metadata.get('next_read'):
                break
            assert page.metadata['next_read'] != arguments
            arguments = page.metadata['next_read']
        assert ''.join(pages) == source
    finally:
        conversation.close()


@pytest.mark.parametrize('interruption', ['timeout', 'cancel'])
def test_native_powershell_interruption_preserves_unterminated_tail(tmp_path, interruption):
    import os
    import sys
    import threading
    import time

    from harness_code_agent.agent.cancellation import CancellationToken, CancelledError
    from harness_code_agent.runtime.builtins.shell import run_bash
    from harness_code_agent.workspace.shell_session import (
        windows_shell_kind,
        windows_shell_path,
    )

    if os.name != 'nt' or windows_shell_kind() != 'pwsh' or not windows_shell_path():
        pytest.skip('requires Windows PowerShell 7')
    source = 'BEFORE_INTERRUPT\n' + 'captured line\n' * 8000 + 'UNTERMINATED_TAIL'
    (tmp_path / 'emit.py').write_text('import sys,time\nsys.stdout.write(' + repr(source) + ')\n'
                                    'sys.stdout.flush()\nopen("ready", "w").close()\ntime.sleep(30)', encoding='utf-8')
    context = make_context(tmp_path, 'native_interrupt')
    conversation = make_conversation(context)
    token = CancellationToken()
    timer = None
    def cancel_when_ready():
        deadline = time.monotonic() + 10
        while not (tmp_path / 'ready').exists() and time.monotonic() < deadline:
            time.sleep(.02)
        token.cancel()
    if interruption == 'cancel':
        timer = threading.Thread(target=cancel_when_ready, daemon=True)
        timer.start()
    try:
        command = "& '" + sys.executable.replace("'", "''") + "' -X utf8 'emit.py'"
        if interruption == 'cancel':
            with pytest.raises(CancelledError) as caught:
                run_bash(command, timeout=15, runtime_state=conversation.runtime_state,
                         tool_context=context, cancellation_token=token)
            result = caught.value.tool_result
            assert result.metadata['cancelled']
        else:
            result = run_bash(command, timeout=2, runtime_state=conversation.runtime_state, tool_context=context)
            assert result.return_code == 130 and result.metadata['timed_out']
        observation = conversation.record_tool_result('native_interrupt', result)
        assert source in observation.raw_output_path.read_text(encoding='utf-8')
        assert not list(tmp_path.rglob('*.raw'))
    finally:
        if timer:
            timer.join(timeout=10)
        conversation.close()


def record_large_result(conversation, call_id='call'):
    conversation.append_message({'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': call_id, 'type': 'function', 'function': {'name': 'test', 'arguments': '{}'}}]})
    return conversation.record_tool_result(call_id, ToolResult(tool='test', status='success', output='the quick brown fox jumps ' * 6000))


def test_fork_resume_restores_structured_reference_and_detects_damage(tmp_path):
    from harness_code_agent.sessions.journal import SessionJournal
    from harness_code_agent.sessions.store import SessionStore

    sessions = SessionStore(tmp_path / '.harness')
    source = sessions.create(profile='coding', cwd=str(tmp_path), model='test', permission_mode='danger-full-access')
    original = make_conversation(make_context(tmp_path, source.id))
    recovered = None
    try:
        observation = record_large_result(original)
        branch = sessions.fork(source.id)
        context = make_context(tmp_path, branch.id)
        recovered = make_conversation(context)
        recovered._replace_messages(SessionJournal(branch.journal_path).recovery_messages('system'), reason='resume')
        reference = observation.artifact_reference
        assert context.tool_output_artifacts[reference['path']]['sha256'] == reference['sha256']
        assert read_file(reference['path'], max_lines=1, tool_context=context).status == 'success'
        observation.raw_output_path.write_text('tampered', encoding='utf-8')
        assert 'damaged' in read_file(reference['path'], max_lines=1, tool_context=context).output
    finally:
        original.close()
        if recovered is not None:
            recovered.close()


def test_worker_copies_only_referenced_artifacts_and_publishes_before_cleanup(tmp_path):
    from harness_code_agent.agent.coordinator import AgentCoordinator, AgentRecord

    (tmp_path / 'note.txt').write_text('code', encoding='utf-8')
    context = make_context(tmp_path, 'session')
    parent = make_conversation(context)
    coordinator = AgentCoordinator(context, parent_messages=lambda: parent.messages,
                                   parent_system_prompt=lambda: parent.agent.system_prompt)
    try:
        inherited = record_large_result(parent)
        unused = '.harness/observations/session/main/missing.txt'
        context.register_output_artifact({'path': unused, 'sha256': 'invalid'})
        record = AgentRecord(id='agent_worker', name='worker', role='worker', task='Inspect note.txt',
                             expected_output='', allowed_paths=['note.txt'], fork_turns='all', model_intensity=None)
        with patch('harness_code_agent.agent.conversation.get_client', return_value=SimpleNamespace(close=lambda: None)):
            coordinator._ensure_conversation(record)
        sandbox = coordinator.changes.sandbox_for(record.id)
        assert (sandbox.workspace / inherited.artifact_reference['path']).is_file()
        assert not (sandbox.workspace / unused).exists()
        produced = record_large_result(record.conversation, 'worker_call')
        record.state = 'completed'
        coordinator._records[record.id] = record
        coordinator._names[record.name] = record.id
        coordinator.close_agent(record.id)
        assert not sandbox.workspace.exists()
        copied = tmp_path / produced.artifact_reference['path']
        assert copied.is_file()
        assert copied.read_text(encoding='utf-8').endswith('the quick brown fox jumps ')
        assert any(entry.kind == 'tool_output_artifact' and entry.payload['path'] == produced.artifact_reference['path']
                   for entry in parent.journal.read())
    finally:
        coordinator.close()
        parent.close()


def test_artifact_write_failure_is_explicit_and_does_not_include_preview(tmp_path):
    store = ObservationStore(tmp_path / 'outputs')
    result = ToolResult(tool='test', status='success', output='SECRET_BODY ' * 10000)
    with patch('pathlib.Path.mkdir', side_effect=OSError('disk full')):
        observation = store.create(tool='test', args={}, result=result, fact_tracker=FactTracker())
    assert 'output_complete: false' in observation.message
    assert 'SECRET_BODY' not in observation.message
    assert observation.raw_output_path is None


def test_intercepted_results_use_the_same_budget(tmp_path):
    from harness_code_agent.agent.tool_executor import ToolExecutor
    from harness_code_agent.runtime.middleware import AgentMiddleware

    class BlockLarge(AgentMiddleware):
        def before_tool(self, *args, **kwargs):
            return ToolResult(tool='read_file', status='failed', output='BLOCKED_BODY ' * 10000)

    conversation = make_conversation(make_context(tmp_path, 'session'))
    conversation.agent.middlewares = [BlockLarge()]
    call = {'id': 'blocked', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"note.txt"}'}}
    conversation.append_message({'role': 'assistant', 'content': None, 'tool_calls': [call]})
    try:
        ToolExecutor(conversation).execute([call])
        message = conversation.messages[-1]['content']
        assert 'BLOCKED_BODY' in message
        assert message.count('BLOCKED_BODY') < 1000
        assert count_tokens([{'role': 'tool', 'content': message}]) <= 5000
        assert 'read_file' in message
        assert conversation.observation_store.observations[-1].raw_output_path.is_file()
    finally:
        conversation.close()


def test_cancelled_shell_keeps_its_captured_result(tmp_path):
    from harness_code_agent.agent.cancellation import CancellationToken, CancelledError
    from harness_code_agent.runtime.builtins.shell import run_bash
    from harness_code_agent.workspace.shell_session import ShellResult

    token = CancellationToken()
    def finish(*args, **kwargs):
        token.cancel()
        return ShellResult(stdout='BEFORE_CANCEL', stderr='', exit_code=130)
    shell = SimpleNamespace(run=finish, close=lambda: None, interrupt=lambda: None)
    with patch('harness_code_agent.workspace.shell_session.PersistentShellSession', return_value=shell), patch('harness_code_agent.runtime.builtins.shell._requires_one_shot_powershell', return_value=False):
        with pytest.raises(CancelledError) as caught:
            run_bash('echo hello', tool_context=make_context(tmp_path, 'session'), cancellation_token=token)
    assert 'BEFORE_CANCEL' in caught.value.tool_result.output
    assert caught.value.tool_result.metadata['cancelled']


def test_shell_log_pages_advance_unicode_offsets_without_duplicate_artifacts(tmp_path):
    from harness_code_agent.agent.observations import observation_dir_for
    from harness_code_agent.runtime.builtins.shell import read_shell_output
    from harness_code_agent.workspace.shell_jobs import ShellJob, ShellOutputBuffer

    context = make_context(tmp_path, 'session')
    source = '中文🙂abc ' * 3000
    buffer = ShellOutputBuffer(observation_dir_for(tmp_path, 'session'))
    buffer.append(source)
    job = ShellJob(job_id='job', command='test', pid=None, process=None, status='exited', output=buffer)
    manager = SimpleNamespace(get=lambda _: job, read_output=lambda _, max_chars, offset: buffer.read(offset, max_chars))
    state = SimpleNamespace(shell_job_manager=manager)
    offset, pages = 0, []
    while offset < len(source):
        result = read_shell_output('job', offset=offset, runtime_state=state, tool_context=context)
        observation, message = observe(tmp_path, '', result=result)
        assert observation.raw_output_path is None
        assert count_tokens([{'role': 'tool', 'content': message}]) <= 5000
        pages.append(result.output.split('\n\n', 1)[1])
        assert result.metadata['next_offset'] > offset
        offset = result.metadata['next_offset']
    assert ''.join(pages) == source
    assert len(list(tmp_path.rglob('*.txt'))) == 1


def test_cancellation_after_tool_completion_preserves_completed_result(tmp_path):
    from harness_code_agent.agent.cancellation import CancellationToken, CancelledError
    from harness_code_agent.agent.tool_executor import ExecutedToolCall, ToolExecutor

    conversation = make_conversation(make_context(tmp_path, 'session'))
    token = CancellationToken()
    call = {'id': 'completed', 'type': 'function', 'function': {'name': 'read_file', 'arguments': '{"path":"note.txt"}'}}
    conversation.append_message({'role': 'assistant', 'content': None, 'tool_calls': [call]})
    executor = ToolExecutor(conversation, token)
    def finish(prepared, cancellation_token):
        token.cancel()
        return ExecutedToolCall(prepared, ToolResult(tool='read_file', status='success', output='COMPLETED_BEFORE_CANCEL'))
    try:
        with patch.object(executor, '_execute_one_unlimited', side_effect=finish):
            with pytest.raises(CancelledError):
                executor.execute([call])
        assert 'COMPLETED_BEFORE_CANCEL' in conversation.messages[-1]['content']
    finally:
        conversation.close()
