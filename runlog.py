"""
runlog.py — one plain-language status report for the whole pipeline.

Every step records what it did into run_report.json, and run_report.txt is rewritten
from it each time, so after any step you can open one file and see, step by step:

    STEP 01  interface analysis
      protein details ............ DEFAULT   'Foo' in proteins.fasta matched no chain
      symmetry search ............ ok        D18, 36 chains, 1 environment
    STEP 03  figures
      fig_packing ................ FAILED    KeyError: 'sc' (traceback in run_errors.log)

Statuses: ok · DEFAULT (ran, but with defaults instead of what was asked) ·
WARNING (ran, read the note) · SKIPPED (not requested, or not applicable) ·
FAILED (that part did not run; the rest of the step carried on).

    R = Report('03', 'figures', model='9o7m.cif')
    with R.guard('fig_packing'):          # a failure here is recorded, not fatal
        ...
    R.add('protein details', 'DEFAULT', 'no FASTA given')
    R.done()
"""
import atexit, json, os, sys, time, traceback

STEPS = {'01': 'interface analysis', '02': 'tables and statistics', '03': 'figures',
         '04': 'PISA concordance', '05': 'assemblies', '06': 'organise results'}
ORDER = ['ok', 'SKIPPED', 'WARNING', 'DEFAULT', 'FAILED']
_JSON, _TXT, _ERR = 'run_report.json', 'run_report.txt', 'run_errors.log'
MANIFEST = 'run_manifest.json'


def _snapshot():
    """{file name: modification time} for the files in the working folder."""
    out = {}
    for f in os.listdir('.'):
        try:
            if os.path.isfile(f):
                out[f] = os.path.getmtime(f)
        except OSError:
            pass
    return out


def read_manifest():
    """{step: [files written by that step]} from the last run in this folder."""
    try:
        return json.load(open(MANIFEST, encoding='utf-8'))
    except Exception:
        return {}


class Skip(Exception):
    """Raise inside a guard to leave it early after RPT.skip(): not an error."""


class _Guard:
    def __init__(self, rep, item, ok_detail, fatal):
        self.rep, self.item, self.ok_detail, self.fatal = rep, item, ok_detail, fatal

    def __enter__(self):
        self.rep.current = self.item
        self.t0 = time.time()
        return self

    def __exit__(self, et, ev, tb):
        self.rep.current = None
        if et is None:
            if self.ok_detail is not None and not self.rep._has(self.item):
                self.rep.add(self.item, 'ok', self.ok_detail)
            return False
        if et in (KeyboardInterrupt, SystemExit):
            return False
        if et is Skip:
            return True
        self.rep.fail(self.item, ev, tb)
        if self.fatal:
            self.rep._fatal_seen = ev      # so the crash hook does not list it twice
        return not self.fatal          # True swallows the error; the step carries on


class Report:
    def __init__(self, step, title=None, model=None):
        self.step = step
        self.current = None
        self.t0 = time.time()
        try:
            self.data = json.load(open(_JSON, encoding='utf-8'))
        except Exception:
            self.data = {}
        if model:
            self.data['_model'] = os.path.basename(model)
        self.data[step] = dict(title=title or STEPS.get(step, ''), items=[],
                               started=time.strftime('%Y-%m-%d %H:%M:%S'),
                               finished=None, seconds=None)
        self._write()
        if step == '01':
            # a new run: start a new list of produced files. Files of an earlier run stay
            # on disk but are no longer counted as part of this run, so step 06 never
            # sweeps them into the results.
            try:
                json.dump({}, open(MANIFEST, 'w', encoding='utf-8'))
            except Exception:
                pass
        self._before = _snapshot()
        atexit.register(self.record_outputs)       # also after a crash or sys.exit
        self._hook = sys.excepthook
        sys.excepthook = self._crash

    # --------------------------------------------------------------- recording
    def _has(self, item):
        return any(i['item'] == item for i in self.data[self.step]['items'])

    def add(self, item, status, detail=''):
        status = status if status in ORDER else 'WARNING'
        it = self.data[self.step]['items']
        it[:] = [i for i in it if i['item'] != item]
        it.append(dict(item=item, status=status, detail=str(detail)))
        self._write()

    def ok(self, item, detail=''):
        self.add(item, 'ok', detail)

    def default(self, item, detail):
        self.add(item, 'DEFAULT', detail)

    def warn(self, item, detail):
        self.add(item, 'WARNING', detail)

    def skip(self, item, detail=''):
        self.add(item, 'SKIPPED', detail)

    def fail(self, item, exc, tb=None):
        msg = f'{type(exc).__name__}: {exc}' if isinstance(exc, BaseException) else str(exc)
        if isinstance(exc, NameError):
            msg = ('not run: it needs the output of an earlier part that failed '
                   f'({msg})')
        msg = msg.replace('\n', ' ')[:220]
        with open(_ERR, 'a', encoding='utf-8') as f:
            f.write(f'\n===== step {self.step} · {item} · {time.ctime()}\n')
            if tb is not None:
                f.write(''.join(traceback.format_exception(type(exc), exc, tb)))
            else:
                f.write(msg + '\n')
        self.add(item, 'FAILED', msg + f'  (traceback in {_ERR})')
        print(f'  !! {item} failed: {msg}', flush=True)

    def guard(self, item, ok_detail='', fatal=False):
        """Context manager: record success, or record the failure and carry on."""
        return _Guard(self, item, ok_detail, fatal)

    def record_outputs(self):
        """Add the files this step created or changed to run_manifest.json."""
        try:
            now = _snapshot()
            new = sorted(f for f, t in now.items()
                         if f not in self._before or t > self._before[f])
            man = read_manifest()
            man[self.step] = sorted(set(man.get(self.step, [])) | set(new))
            tmp = MANIFEST + '.tmp'
            json.dump(man, open(tmp, 'w', encoding='utf-8'), indent=1)
            os.replace(tmp, MANIFEST)
        except Exception:
            pass                                   # bookkeeping must never stop a run

    def done(self):
        d = self.data[self.step]
        d['finished'] = time.strftime('%Y-%m-%d %H:%M:%S')
        d['seconds'] = round(time.time() - self.t0, 1)
        self._write()
        self.record_outputs()
        sys.excepthook = self._hook
        n = {s: sum(1 for i in d['items'] if i['status'] == s) for s in ORDER}
        bad = n['FAILED'] + n['DEFAULT'] + n['WARNING']
        print(f'run report: {n["ok"]} ok, {n["DEFAULT"]} default, {n["WARNING"]} warning, '
              f'{n["FAILED"]} failed -> {_TXT}' if bad else
              f'run report: all {n["ok"]} items ok -> {_TXT}', flush=True)

    def _crash(self, et, ev, tb):
        """An error nothing caught: record WHERE it happened before the step dies."""
        try:
            if getattr(self, '_fatal_seen', None) is not ev:
                self.fail(self.current or 'step aborted', ev, tb)
            self.data[self.step]['finished'] = 'ABORTED ' + time.strftime('%H:%M:%S')
            self._write()
        finally:
            self._hook(et, ev, tb)

    # --------------------------------------------------------------- writing
    def _write(self):
        try:
            tmp = _JSON + '.tmp'
            json.dump(self.data, open(tmp, 'w', encoding='utf-8'), indent=1)
            os.replace(tmp, _JSON)
            open(_TXT, 'w', encoding='utf-8').write(render(self.data))
        except Exception:
            pass                           # the report must never be what kills a run


def render(data):
    L = ['Interface analysis — run report', '']
    if data.get('_model'):
        L.append(f'model: {data["_model"]}')
    L.append(f'written: {time.strftime("%Y-%m-%d %H:%M:%S")}')
    L.append('')
    tot = {s: 0 for s in ORDER}
    for step in sorted(k for k in data if not k.startswith('_')):
        d = data[step]
        fin = d.get('finished')
        when = (f'finished in {d["seconds"]:.0f} s' if d.get('seconds') is not None
                else (fin if fin else 'running or interrupted'))
        L.append(f'STEP {step}  {d.get("title", "")}'.ljust(52) + when)
        for i in d['items']:
            tot[i['status']] += 1
            name = (i['item'] + ' ').ljust(30, '.')
            L.append(f'  {name} {i["status"]:<8}  {i["detail"]}')
        L.append('')
    L.append('summary: ' + ', '.join(f'{tot[s]} {s.lower() if s == "ok" else s}'
                                     for s in ORDER if tot[s]))
    L.append('')
    L.append('ok       ran as asked')
    L.append('DEFAULT  ran, but with defaults instead of what was asked — see the note')
    L.append('WARNING  ran; read the note before relying on that output')
    L.append('SKIPPED  not requested, or not applicable to this model')
    L.append('FAILED   that part did not run; everything else in the step still did')
    return '\n'.join(L) + '\n'
