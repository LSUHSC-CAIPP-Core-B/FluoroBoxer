"""Exercise the standalone notebook without copying the user's dataset."""
import json
from pathlib import Path

import pytest


@pytest.fixture
def notebook():
    path = Path(__file__).resolve().parents[1] / 'biological_train_validation_test_split.ipynb'
    document = json.loads(path.read_text())
    scope = {}
    # Load imports/config and function definitions, leaving execution to each test.
    import ast
    for cell in document['cells']:
        if cell['cell_type'] != 'code':
            continue
        tree = ast.parse(''.join(cell['source']))
        if not scope:
            exec(compile(tree, str(path), 'exec'), scope)
        else:
            definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
            exec(compile(ast.Module(body=definitions, type_ignores=[]), str(path), 'exec'), scope)
    return scope


def make_source(root):
    for well in ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']:
        for field in [1, 2]:
            for time in ['00d00h00m', '00d02h00m']:
                for channel in ['phase', 'Green']:
                    folder = root / 'Batch1' / 'Apoptosis' / ('MEF_' + channel)
                    folder.mkdir(parents=True, exist_ok=True)
                    (folder / ('VID1_{}_{}_{}.png'.format(well, field, time))).write_bytes(b'image')


def test_wells_stay_together_and_export(notebook, tmp_path):
    source, output = tmp_path / 'raw', tmp_path / 'split'
    make_source(source)
    pairs, missing = notebook['discover_pairs'](source)
    ratios = {'train': .8, 'val': .1, 'test': .1}
    manifest = notebook['assign_splits'](pairs, 'well', ratios, 99)
    repeated = notebook['assign_splits'](list(reversed(pairs)), 'well', ratios, 99)
    assert {r['phase']: r['split'] for r in manifest} == {r['phase']: r['split'] for r in repeated}
    assert len(pairs) == 24 and not missing
    for well in {row['well'] for row in manifest}:
        assert len({row['split'] for row in manifest if row['well'] == well}) == 1
    assert {split: sum(r['split'] == split for r in manifest) for split in ratios} == {'train': 16, 'val': 4, 'test': 4}
    notebook['export_split'](source, output, manifest, missing, {})
    assert len(list(output.rglob('*.png'))) == 48
    assert len(list(source.rglob('*.png'))) == 48
    assert (output / 'manifest.csv').is_file()
    assert list(output.glob('*/Batch1/Apoptosis/MEF_Phase/*.png'))
    with pytest.raises(FileExistsError):
        notebook['export_split'](source, output, manifest, missing, {})


def test_missing_partner_and_bad_name(notebook, tmp_path):
    make_source(tmp_path)
    next(tmp_path.rglob('MEF_Green/*.png')).unlink()
    pairs, missing = notebook['discover_pairs'](tmp_path)
    assert len(pairs) == 23 and len(missing) == 1
    with pytest.raises(ValueError, match='unmatched'):
        notebook['discover_pairs'](tmp_path, 'error')
    for channel in ['phase', 'Green']:
        (tmp_path / 'Batch1' / 'Apoptosis' / ('MEF_' + channel) / 'unknown.png').touch()
    with pytest.raises(ValueError, match='Unexpected raw filename'):
        notebook['discover_pairs'](tmp_path)


def test_batch_split_and_invalid_ratios(notebook):
    rows = [dict(batch=batch, video='VID1', well='A1', death_type='Apoptosis', cell_type='MEF')
            for batch in ['Batch1', 'Batch2']]
    with pytest.raises(ValueError, match='cannot fill'):
        notebook['assign_splits'](rows, 'batch', {'train': .8, 'val': .1, 'test': .1}, 99)
    result = notebook['assign_splits'](rows, 'batch', {'train': .5, 'val': .5, 'test': 0}, 99)
    assert {r['split'] for r in result} == {'train', 'val'}
    for ratios in [{'train': .5}, {'train': 1.1, 'test': -.1}, {'train': float('nan')}]:
        with pytest.raises(ValueError):
            notebook['allocate_counts'](6, ratios)
