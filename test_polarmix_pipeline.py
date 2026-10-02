import os
import sys
import numpy as np
import pytest
from PIL import Image
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
from dataloader.polarmix.labels import LabelMap, encode_instances, decode_instances
from dataloader.polarmix.instances import (
    Instance,
    InstanceBank,
    extract_instances,
    rotate_about_z,
    rotate_copy_instances,
)
from dataloader.polarmix.mix import sector_mask, swap_sectors, polar_mix
from dataloader.polarmix.weak import generate_weak_labels, make_weak_labels
from dataloader.polarmix.io import read_points, read_labels, read_scan
class StubRNG:
    '''Deterministic stand-in for numpy Generator used by polar_mix.'''
    def __init__(self, values):
        self.values = list(values)
    def random(self):
        return self.values.pop(0)
def test_label_map_remap_known_unknown_and_edges():
    mapping = os.path.join(REPO_ROOT, 'config', 'label_mapping', 'semantic-kitti-all.yaml')
    label_map = LabelMap(mapping)
    assert label_map.ignore_label == 0
    raw = np.array([10, 0, 999, 40, 255], dtype=np.int64)
    remapped = label_map.remap(raw)
    assert remapped.tolist() == [1, 0, 0, 9, 8]
    empty = label_map.remap(np.array([], dtype=np.int64))
    assert empty.shape == (0,)
    raw_2d = np.array([[10, 40], [0, 1]])
    assert label_map.remap(raw_2d).tolist() == [[1, 9], [0, 0]]
    assert raw.tolist() == [10, 0, 999, 40, 255]
def test_encode_decode_roundtrip():
    sem = np.array([1, 2, 255, 65535])
    inst = np.array([0, 1, 65535, 3])
    packed = encode_instances(sem, inst)
    assert packed.dtype == np.uint32
    assert packed.tolist() == [
        (0 << 16) | 1, (1 << 16) | 2, (65535 << 16) | 255, (3 << 16) | 65535,
    ]
    out_sem, out_inst = decode_instances(packed)
    assert out_sem.tolist() == sem.tolist()
    assert out_inst.tolist() == inst.tolist()
def test_encode_errors():
    with pytest.raises(ValueError) as err:
        encode_instances([1, 2], [1, 2, 3])
    assert str(err.value) == 'sem_labels and instance_labels must have the same shape'
    with pytest.raises(ValueError) as err:
        encode_instances([-1], [0])
    assert str(err.value) == 'semantic labels must fit in 16 bits'
    with pytest.raises(ValueError) as err:
        encode_instances([70000], [0])
    assert str(err.value) == 'semantic labels must fit in 16 bits'
    with pytest.raises(ValueError) as err:
        encode_instances([1], [-1])
    assert str(err.value) == 'instance labels must fit in 16 bits'
    with pytest.raises(ValueError) as err:
        encode_instances([1], [70000])
    assert str(err.value) == 'instance labels must fit in 16 bits'
def test_extract_instances_filters_groups_and_order():
    points = np.array([
        [0, 0, 0, 0.1],   # road, no instance -> dropped
        [1, 0, 0, 0.2],   # car 4
        [2, 0, 0, 0.3],   # car 4
        [3, 0, 0, 0.4],   # bicycle 7
        [4, 0, 0, 0.5],   # car 2 (single point)
        [5, 0, 0, 0.6],   # unlabeled -> dropped
        [6, 0, 0, 0.7],   # bicycle 7
        [7, 0, 0, 0.8],   # bicycle 7
    ], dtype=np.float32)
    sem = np.array([9, 1, 1, 2, 1, 0, 2, 2])
    inst = np.array([0, 4, 4, 7, 2, 5, 7, 7])
    instances = extract_instances(points, sem, inst)
    assert [(x.sem_label, x.instance_id, len(x.points)) for x in instances] == [
        (1, 2, 1), (1, 4, 2), (2, 7, 3),
    ]
    assert np.allclose(instances[0].points, [[4, 0, 0, 0.5]])
    instances = extract_instances(points, sem, inst, min_points=2)
    assert [(x.sem_label, x.instance_id) for x in instances] == [(1, 4), (2, 7)]
    assert len(extract_instances(points, sem, inst, min_points=0)) == 3
    only_car = extract_instances(points, sem, inst, instance_classes=(1,))
    assert all(x.sem_label == 1 for x in only_car)
    assert len(only_car) == 2
    grouped = extract_instances(points, sem, inst)[1]
    grouped.points[:] = -99
    assert np.allclose(points[1], [1, 0, 0, 0.2])
    assert extract_instances(np.zeros((0, 4), np.float32), [], []) == []
    with pytest.raises(ValueError) as err:
        extract_instances(points, sem, inst[:-1])
    assert str(err.value) == 'points, sem_labels and instance_labels must have the same length'
def test_rotate_about_z_values_and_dtype():
    points = np.array([[1, 0, 0, 7], [0, 2, 5, 8]], dtype=np.float32)
    out = rotate_about_z(points, np.pi / 2)
    assert out.dtype == np.float32
    assert np.allclose(out, [[0, 1, 0, 7], [-2, 0, 5, 8]], atol=1e-6)
    out = rotate_about_z(points, 0.0)
    assert np.allclose(out, points)
    assert out is not points
    out = rotate_about_z(points, -np.pi / 2)
    assert np.allclose(out, [[0, -1, 0, 7], [2, 0, 5, 8]], atol=1e-6)
    assert points.tolist() == [[1.0, 0.0, 0.0, 7.0], [0.0, 2.0, 5.0, 8.0]]
    three_col = rotate_about_z(np.array([[1, 0, 5]], dtype=np.float32), np.pi)
    assert three_col.shape == (1, 3)
    assert np.allclose(three_col, [[-1, 0, 5]], atol=1e-6)
def test_rotate_copy_instances_order_and_empty():
    insts = [
        Instance(points=np.array([[1, 0, 0, 1]], np.float32), sem_label=1, instance_id=3),
        Instance(points=np.array([[0, 1, 0, 2], [1, 1, 0, 2]], np.float32),
                 sem_label=2, instance_id=4),
    ]
    points, sem, inst = rotate_copy_instances(insts, [np.pi / 2])
    assert np.allclose(points, [
        [1, 0, 0, 1],
        [0, 1, 0, 2], [1, 1, 0, 2],
        [0, 1, 0, 1],
        [-1, 0, 0, 2], [-1, 1, 0, 2],
    ], atol=1e-6)
    assert sem.tolist() == [1, 2, 2, 1, 2, 2]
    assert inst.tolist() == [3, 4, 4, 3, 4, 4]
    points, sem, inst = rotate_copy_instances(insts, [])
    assert len(points) == 3 and sem.tolist() == [1, 2, 2]
    points, sem, inst = rotate_copy_instances([], [np.pi])
    assert points.shape[0] == 0 and sem.shape == (0,) and inst.shape == (0,)
def test_sector_mask_boundaries_wrap_and_zero_width():
    assert sector_mask([0.0, 0.5, 1.0], 0.0, 1.0).tolist() == [False, True, False]
    assert sector_mask([2.0, -2.0, 0.0], 1.5, -1.0).tolist() == [True, True, False]
    assert sector_mask([0.1, 0.2], 0.3, 0.3).tolist() == [False, False]
    assert sector_mask([0.1, 0.2], 0.3, 0.3 + 2 * np.pi).tolist() == [False, False]
    assert sector_mask([], 0.0, 1.0).shape == (0,)
def test_swap_sectors_exact_and_immutable():
    points_a = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0]], dtype=np.float32)
    labels_a = np.array([10, 20, 30])
    points_b = np.array([[5, 5, 0], [0, -5, 0], [3, 3, 0]], dtype=np.float32)
    labels_b = np.array([40, 50, 60])
    out_a, out_a_labels, out_b, out_b_labels = swap_sectors(
        points_a, labels_a, points_b, labels_b, -np.pi / 2, np.pi / 2)
    assert np.allclose(out_a, [[0, 1, 0], [-1, 0, 0], [5, 5, 0], [3, 3, 0]])
    assert out_a_labels.tolist() == [20, 30, 40, 60]
    assert np.allclose(out_b, [[0, -5, 0], [1, 0, 0]])
    assert out_b_labels.tolist() == [50, 10]
    out_a, out_a_labels, out_b, out_b_labels = swap_sectors(
        points_a, labels_a, points_b, labels_b, 0.3, 0.3)
    assert np.allclose(out_a, points_a) and np.allclose(out_b, points_b)
    assert out_a_labels.tolist() == [10, 20, 30]
    assert out_b_labels.tolist() == [40, 50, 60]
    assert points_a.tolist() == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    with pytest.raises(ValueError) as err:
        swap_sectors(points_a, labels_a[:-1], points_b, labels_b, -1.0, 1.0)
    assert str(err.value) == 'points and labels must have the same length'
def test_polar_mix_flags_and_determinism():
    points_a = np.array([[1, 0, 0, 0.1]], np.float32)
    sem_a = np.array([9])
    inst_a = np.array([0])
    points_b = np.array([[2, 0, 0, 0.2], [0, 2, 0, 0.3]], np.float32)
    sem_b = np.array([1, 1])
    inst_b = np.array([7, 7])
    out = polar_mix(points_a, sem_a, inst_a, points_b, sem_b, inst_b,
                    rng=StubRNG([0.9, 0.9]),
                    swap_probability=0.0, rotate_probability=0.0)
    assert np.allclose(out['points'], points_a)
    assert out['sem_labels'].tolist() == [9]
    assert out['instance_labels'].tolist() == [0]
    expected = swap_sectors(points_a, sem_a, points_b, sem_b, -np.pi / 2, np.pi / 2)
    out = polar_mix(points_a, sem_a, inst_a, points_b, sem_b, inst_b,
                    rng=StubRNG([0.0, 0.9]), swap_probability=1.0,
                    rotate_probability=0.0, alpha=-np.pi / 2)
    exp_points, exp_sem, exp_pts_b, exp_sem_b = expected
    _, exp_inst, _, _ = swap_sectors(points_a, inst_a, points_b, inst_b,
                                     -np.pi / 2, np.pi / 2)
    assert np.allclose(out['points'], exp_points)
    assert out['sem_labels'].tolist() == exp_sem.tolist()
    assert out['instance_labels'].tolist() == exp_inst.tolist()
    from dataloader.polarmix.instances import rotate_copy_instances
    copies = rotate_copy_instances(
        extract_instances(points_b, sem_b, inst_b), [np.pi / 2])
    out = polar_mix(points_a, sem_a, inst_a, points_b, sem_b, inst_b,
                    rng=StubRNG([0.9, 0.5]),
                    swap_probability=0.0, rotate_probability=1.0, omegas=[np.pi / 2])
    assert np.allclose(out['points'], np.concatenate([points_a, copies[0]], axis=0))
    assert out['sem_labels'].tolist() == [9, 1, 1, 1, 1]
    assert out['instance_labels'].tolist() == [0, 7, 7, 7, 7]
    assert np.array_equal(points_a, np.array([[1, 0, 0, 0.1]], np.float32))
    out = polar_mix(points_a, sem_a, inst_a,
                    np.zeros((0, 4), np.float32), [], [],
                    rng=StubRNG([0.9, 0.5]), swap_probability=0.0,
                    rotate_probability=1.0, omegas=[np.pi])
    assert np.allclose(out['points'], points_a)
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError) as err:
            polar_mix(points_a, sem_a, inst_a, points_b, sem_b, inst_b,
                      rng=StubRNG([0.0, 0.0]), swap_probability=bad)
        assert str(err.value) == 'probabilities must be in [0, 1]'
def test_polar_mix_draws_alpha_between_decisions():
    points_a = np.array([[1, 0, 0, 0.1], [0, 1, 0, 0.2]], np.float32)
    sem_a = np.array([9, 9])
    inst_a = np.array([0, 0])
    points_b = np.array([[2, 0, 0, 0.3], [0, 2, 0, 0.4]], np.float32)
    sem_b = np.array([1, 1])
    inst_b = np.array([7, 7])
    out = polar_mix(points_a, sem_a, inst_a, points_b, sem_b, inst_b,
                    rng=StubRNG([0.0, 0.5, 0.0]),
                    swap_probability=0.5, rotate_probability=1.0, omegas=[])
    swapped = swap_sectors(points_a, sem_a, points_b, sem_b, -np.pi / 2, np.pi / 2)
    paste_pts, paste_sem, paste_inst = rotate_copy_instances(
        extract_instances(points_b, sem_b, inst_b), [])
    expected_points = np.concatenate([swapped[0], paste_pts], axis=0)
    expected_sem = np.concatenate([swapped[1], paste_sem])
    assert np.allclose(out['points'], expected_points)
    assert out['sem_labels'].tolist() == expected_sem.tolist()
    assert out['instance_labels'].tolist() == [0, 7, 7, 7]
def test_make_weak_labels_floor_count_and_instance_bits():
    packed = encode_instances([5, 5, 5, 5], [1, 2, 3, 4])
    out = make_weak_labels(packed, 0.5, rng=np.random.default_rng(7))
    assert out.dtype == np.uint32
    low = (out & np.uint32(0xFFFF)).astype(np.int64)
    high = (out >> np.uint32(16)).astype(np.int64)
    assert int((low == 0).sum()) == 2
    assert high.tolist() == [1, 2, 3, 4]
    assert packed.tolist() == encode_instances([5, 5, 5, 5], [1, 2, 3, 4]).tolist()
    packed5 = encode_instances([5] * 5, [1] * 5)
    out5 = make_weak_labels(packed5, 0.5, rng=np.random.default_rng(7))
    assert int(((out5 & np.uint32(0xFFFF)) == 0).sum()) == 2
    assert np.array_equal(make_weak_labels(packed, 0.0), packed)
    all_masked = make_weak_labels(packed, 1.0, rng=np.random.default_rng(7))
    assert ((all_masked & np.uint32(0xFFFF)) == 0).all()
    assert ((all_masked >> np.uint32(16)) == np.array([1, 2, 3, 4], np.uint32)).all()
    empty = make_weak_labels(np.array([], np.uint32), 0.5)
    assert empty.shape == (0,)
def test_make_weak_labels_rejects_bad_fraction():
    packed = encode_instances([5], [1])
    for bad in (-0.01, 1.01):
        with pytest.raises(ValueError) as err:
            make_weak_labels(packed, bad)
        assert str(err.value) == 'fraction must be in [0, 1]'
def _write_scan(root, stem, points, packed):
    velo = os.path.join(root, 'velodyne')
    labels = os.path.join(root, 'labels')
    os.makedirs(velo, exist_ok=True)
    os.makedirs(labels, exist_ok=True)
    points.astype(np.float32).tofile(os.path.join(velo, stem + '.bin'))
    packed.astype(np.uint32).tofile(os.path.join(labels, stem + '.label'))
def test_scan_io_roundtrip_missing_labels_and_errors(tmp_path):
    root = str(tmp_path)
    points = np.arange(12, dtype=np.float32).reshape(3, 4)
    packed = encode_instances([10, 11, 40], [1, 2, 0])
    _write_scan(root, '000000', points, packed)
    assert read_points(os.path.join(root, 'velodyne', '000000.bin')).shape == (3, 4)
    assert read_labels(os.path.join(root, 'labels', '000000.label')).tolist() == packed.tolist()
    got_points, got_labels = read_scan(os.path.join(root, 'velodyne', '000000.bin'))
    assert np.allclose(got_points, points)
    assert got_labels.tolist() == packed.tolist()
    points2 = np.ones((2, 4), np.float32)
    points2.tofile(os.path.join(root, 'velodyne', '000002.bin'))
    got_points, got_labels = read_scan(os.path.join(root, 'velodyne', '000002.bin'))
    assert got_labels.tolist() == [0, 0]
    wrong = np.zeros(2, np.uint32)
    wrong.tofile(os.path.join(root, 'labels', '000000.label'))
    with pytest.raises(ValueError) as err:
        read_scan(os.path.join(root, 'velodyne', '000000.bin'))
    assert str(err.value) == 'Scan and Label don' + chr(39) + 't contain same number of points'
    np.zeros(0, np.float32).tofile(os.path.join(root, 'velodyne', '000003.bin'))
    got_points, got_labels = read_scan(os.path.join(root, 'velodyne', '000003.bin'))
    assert got_points.shape == (0, 4) and got_labels.shape == (0,)
MINI_YAML = '''
labels:
  0: 'unlabeled'
  1: 'outlier'
  10: 'car'
  11: 'bicycle'
  40: 'road'
  50: 'building'
learning_map:
  0: 0
  1: 0
  10: 1
  11: 2
  40: 9
  50: 13
split:
  train: [0]
  valid: [0]
  test: [0]
'''
CALIB = ('P2: 100 0 50 0 0 100 25 0 0 0 1 0' + chr(10)
         + 'Tr: 1 0 0 0 0 1 0 0 0 0 1 0' + chr(10))
A_POINTS = np.array([[1, 0, 0, 0.1], [2, 0, 0, 0.2], [3, 0, 0, 0.3],
                     [4, 0, 0, 0.4], [10, 0, 0, 0.5], [11, 0, 0, 0.6]], np.float32)
A_SEM = np.array([40, 40, 40, 40, 10, 10])
A_INST = np.array([0, 0, 0, 0, 1, 1])
B_POINTS = np.array([[1, 5, 0, 0.1], [2, 5, 0, 0.2], [3, 5, 0, 0.3],
                     [20, 5, 0, 0.4], [21, 5, 0, 0.5]], np.float32)
B_SEM = np.array([40, 40, 40, 11, 11])
B_INST = np.array([0, 0, 0, 5, 5])
def _pack(sem, inst):
    return (sem.astype(np.uint32) | (inst.astype(np.uint32) << np.uint32(16)))
def _build_mini_kitti(tmp_path):
    root = tmp_path / '00'
    _write_scan(str(root), '000000', A_POINTS, _pack(A_SEM, A_INST))
    _write_scan(str(root), '000001', B_POINTS, _pack(B_SEM, B_INST))
    (root / 'calib.txt').write_text(CALIB)
    (root / 'image_2').mkdir(exist_ok=True)
    for stem in ('000000', '000001'):
        Image.new('RGB', (8, 8), (0, 0, 0)).save(str(root / 'image_2' / (stem + '.png')))
    return tmp_path
def _mini_config(tmp_path, **overrides):
    mapping = tmp_path / 'mini_labels.yaml'
    mapping.write_text(MINI_YAML)
    cfg = {'dataset_params': {
        'label_mapping': str(mapping),
        'ignore_label': 0,
        'polar_mix': True,
        'swap_probability': 0.0,
        'rotate_probability': 1.0,
        'omegas': [0.0],
        'debug': False,
    }}
    cfg['dataset_params'].update(overrides)
    return cfg
def test_reader_is_registered():
    import dataloader.pc_dataset as pc_dataset
    from dataloader.polarmix.dataset import SemanticKITTI_PolarMix
    assert pc_dataset.get_pc_model_class('SemanticKITTI_polarmix_1') is SemanticKITTI_PolarMix
    assert 'SemanticKITTI_polarmix' in pc_dataset.REGISTERED_PC_DATASET_CLASSES
def test_train_reader_mixes_scans(tmp_path):
    from dataloader.polarmix.dataset import SemanticKITTI_PolarMix
    _build_mini_kitti(tmp_path)
    cfg = _mini_config(tmp_path)
    dataset = SemanticKITTI_PolarMix(cfg, str(tmp_path), imageset='train',
                                     rng=np.random.default_rng(3))
    assert len(dataset) == 2
    data, path = dataset[0]
    assert path.endswith('000000.bin')
    assert data['origin_len'] == 6
    assert len(data['xyz']) == 10          # 6 base + 2 instance points x 2 copies
    assert data['xyz'].dtype == np.float32
    assert np.allclose(data['xyz'][:6], A_POINTS[:, :3])
    assert data['labels'].tolist()[:6] == [9, 9, 9, 9, 1, 1]
    assert data['labels_o'].tolist()[:6] == [40, 40, 40, 40, 10, 10]
    assert data['signal'].shape == (10, 1)
    tail = data['xyz'][6:]
    expected_b = np.concatenate([B_POINTS[3:, :3], B_POINTS[3:, :3]])
    expected_a = np.concatenate([A_POINTS[4:, :3], A_POINTS[4:, :3]])
    assert np.allclose(tail, expected_b) or np.allclose(tail, expected_a)
    assert set(data['labels'][6:].tolist()) <= {1, 2}
    assert data['instance_label'].shape == (10,)
    assert len(set(data['instance_label'][6:].tolist())) == 1
    assert int(data['instance_label'][6]) in (1, 5)
    p2 = np.array([[100, 0, 50, 0], [0, 100, 25, 0], [0, 0, 1, 0]], np.float64)
    tr = np.eye(4)
    tr[:3, :4] = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]], np.float64)
    assert np.allclose(data['proj_matrix'], p2 @ tr)
    assert data['img'].size == (8, 8)
def test_val_and_test_readers_do_not_mix(tmp_path):
    from dataloader.polarmix.dataset import SemanticKITTI_PolarMix
    _build_mini_kitti(tmp_path)
    cfg = _mini_config(tmp_path)
    val = SemanticKITTI_PolarMix(cfg, str(tmp_path), imageset='val',
                                 rng=np.random.default_rng(3))
    data, path = val[1]
    assert path.endswith('000001.bin')
    assert data['origin_len'] == 5 and len(data['xyz']) == 5
    assert np.allclose(data['xyz'], B_POINTS[:, :3])
    assert data['labels'].tolist() == [9, 9, 9, 2, 2]
    assert data['labels_o'].tolist() == [40, 40, 40, 11, 11]
    test = SemanticKITTI_PolarMix(cfg, str(tmp_path), imageset='test',
                                  rng=np.random.default_rng(3))
    data, _ = test[0]
    assert len(data['xyz']) == 6
    assert data['labels'].tolist() == [0] * 6
    with pytest.raises(Exception) as err:
        SemanticKITTI_PolarMix(cfg, str(tmp_path), imageset='whatever')
    assert str(err.value) == 'Split must be train/val/test'
def test_num_vote_repeats_files(tmp_path):
    from dataloader.polarmix.dataset import SemanticKITTI_PolarMix
    _build_mini_kitti(tmp_path)
    cfg = _mini_config(tmp_path)
    dataset = SemanticKITTI_PolarMix(cfg, str(tmp_path), imageset='val', num_vote=3)
    assert len(dataset) == 6
    paths = [dataset[i][1] for i in range(6)]
    assert sum(p.endswith('000000.bin') for p in paths) == 3
    assert sum(p.endswith('000001.bin') for p in paths) == 3
def test_instance_bank_from_scans_counts(tmp_path):
    _build_mini_kitti(tmp_path)
    cfg = _mini_config(tmp_path)
    label_map = LabelMap(cfg['dataset_params']['label_mapping'])
    paths = [str(tmp_path / '00' / 'velodyne' / '000000.bin'),
             str(tmp_path / '00' / 'velodyne' / '000001.bin')]
    bank = InstanceBank.from_scans(paths, label_map)
    assert set(bank.classes) == {1, 2}
    assert bank.count(1) == 1 and bank.count(2) == 1
    assert bank.count(99) == 0
    assert bank.total_instances == 2
    instance = bank.sample_many(1, [1, 2], p=1.0, rng=np.random.default_rng(0))[0]
    assert instance.sem_label in (1, 2)
    assert instance.instance_id in (1, 5)
def test_instance_bank_class_balance_and_p():
    rare = Instance(np.zeros((1, 4), np.float32), 1, 1)
    bank = InstanceBank({1: [rare]})
    bank.instances_by_class[2] = [
        Instance(np.zeros((1, 4), np.float32), 2, 100 + i) for i in range(50)
    ]
    rng = np.random.default_rng(0)
    draws = [bank.sample_many(1, [1, 2], p=1.0, rng=rng)[0].sem_label
             for _ in range(200)]
    assert draws.count(1) > 150
    rng0 = np.random.default_rng(0)
    draws0 = [bank.sample_many(1, [1, 2], p=0.0, rng=rng0)[0].sem_label
              for _ in range(200)]
    assert 60 <= draws0.count(1) <= 140
def test_instance_bank_edges():
    empty = InstanceBank()
    assert list(empty.classes) == [] and empty.total_instances == 0
    assert empty.sample_many(0, [1]) == []
    with pytest.raises(ValueError) as err:
        empty.sample_many(1, [1])
    assert str(err.value) == 'instance bank is empty'
    with pytest.raises(ValueError) as err:
        empty.sample_many(2, [1])
    assert str(err.value) == 'instance bank is empty'
    only = Instance(np.zeros((1, 4), np.float32), 3, 7)
    bank = InstanceBank({3: [only]})
    assert bank.sample_many(1, [3], rng=np.random.default_rng(0))[0] is only
    with pytest.raises(ValueError) as err:
        bank.sample_many(1, [3], p=-0.5)
    assert str(err.value) == 'p must be non-negative'
    assert bank.sample_many(1, [1, 3], rng=np.random.default_rng(0))[0] is only
def test_polar_mix_uses_instance_bank():
    points_a = np.array([[1, 0, 0, 0.1]], np.float32)
    sem_a, inst_a = np.array([9]), np.array([0])
    points_b = np.array([[2, 0, 0, 0.2]], np.float32)
    sem_b, inst_b = np.array([2]), np.array([40])
    bank_instance = Instance(
        points=np.array([[5, 0, 0, 0.5], [6, 0, 0, 0.6]], np.float32),
        sem_label=1, instance_id=9)
    bank = InstanceBank({1: [bank_instance]})
    out = polar_mix(points_a, sem_a, inst_a, points_b, sem_b, inst_b,
                    rng=np.random.default_rng(0),
                    swap_probability=0.0, rotate_probability=1.0, omegas=[0.0],
                    instance_bank=bank, num_paste=2)
    expected = rotate_copy_instances([bank_instance, bank_instance], [0.0])
    assert np.allclose(out['points'], np.concatenate([points_a, expected[0]]))
    assert out['sem_labels'].tolist() == [9] + [1] * 8
    assert out['instance_labels'].tolist() == [0] + [9] * 8
def test_polar_mix_bank_empty_only_when_pasting():
    points_a = np.array([[1, 0, 0, 0.1]], np.float32)
    sem_a, inst_a = np.array([9]), np.array([0])
    points_b = np.array([[2, 0, 0, 0.2]], np.float32)
    sem_b, inst_b = np.array([2]), np.array([40])
    out = polar_mix(points_a, sem_a, inst_a, points_b, sem_b, inst_b,
                    rng=StubRNG([0.9, 0.9]), swap_probability=0.0,
                    rotate_probability=0.0, instance_bank=InstanceBank())
    assert np.allclose(out['points'], points_a)
    with pytest.raises(ValueError) as err:
        polar_mix(points_a, sem_a, inst_a, points_b, sem_b, inst_b,
                  rng=StubRNG([0.9, 0.5]), swap_probability=0.0,
                  rotate_probability=1.0, instance_bank=InstanceBank())
    assert str(err.value) == 'instance bank is empty'
    bank = InstanceBank({1: [Instance(points_a.copy(), 1, 3)]})
    with pytest.raises(ValueError) as err:
        polar_mix(points_a, sem_a, inst_a, points_b, sem_b, inst_b,
                  rng=StubRNG([0.9, 0.5]), swap_probability=0.0,
                  rotate_probability=1.0, instance_bank=bank, repaste_p=-1.0)
    assert str(err.value) == 'p must be non-negative'
def test_generate_weak_labels_writes_files(tmp_path):
    _build_mini_kitti(tmp_path)
    written = generate_weak_labels(str(tmp_path), 0.5, rng=np.random.default_rng(0))
    assert written == 2
    out_a = read_labels(str(tmp_path / '00' / 'labels_w_0.5' / '000000.label'))
    assert out_a.shape == (6,)
    assert int(((out_a & np.uint32(0xFFFF)) == 0).sum()) == 3  # floor(0.5 * 6)
    assert (out_a >> np.uint32(16)).tolist() == A_INST.tolist()
    out_b = read_labels(str(tmp_path / '00' / 'labels_w_0.5' / '000001.label'))
    assert int(((out_b & np.uint32(0xFFFF)) == 0).sum()) == 2  # floor(0.5 * 5)
    src_a = read_labels(str(tmp_path / '00' / 'labels' / '000000.label'))
    assert (src_a & np.uint32(0xFFFF)).tolist() == A_SEM.tolist()
    assert generate_weak_labels(str(tmp_path), 0.5, rng=np.random.default_rng(1)) == 2
    assert generate_weak_labels(str(tmp_path / 'does_not_exist'), 0.5) == 0
