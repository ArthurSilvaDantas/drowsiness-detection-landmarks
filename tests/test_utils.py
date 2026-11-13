import numpy as np
import pytest

import utils

DATA = "data"


def test_crop_matches_reference_example():
    # reference example: box (692, 90)-(854, 326); landmark 60 (720.6, 190.7) -> (82.5, 113.0)
    m = utils.crop_matrix([692, 90, 854, 326], 256, 1.25)
    p = utils.transform_points(np.array([[720.6, 190.7]]), m)[0]
    assert np.allclose(p, [82.5, 113.0], atol=0.1)


def test_crop_roundtrip():
    rng = np.random.default_rng(0)
    pts = rng.uniform(0, 1000, size=(98, 2))
    m = utils.crop_matrix([100, 200, 400, 600], 256)
    back = utils.transform_points(utils.transform_points(pts, m), utils.invert(m))
    assert np.abs(back - pts).max() < 1e-9


def test_augment_identity_equals_plain_crop():
    m = utils.augment_matrix((192, 192), 1.0, 256)
    p = utils.transform_points(np.array([[192.0, 192.0]]), m)[0]
    assert np.allclose(p, [128, 128])


@pytest.fixture(scope="module")
def mirror():
    return utils.read_mirror(DATA)


def test_mirror_is_involution(mirror):
    assert (mirror[mirror] == np.arange(98)).all()
    assert mirror[60] == 72 and mirror[0] == 32 and mirror[96] == 97


def test_flip_twice_is_identity_and_keeps_eye_side(mirror):
    face = utils.read_faces(DATA, "train")[0]
    m = utils.crop_matrix(face.box, 256)
    pts = utils.transform_points(face.landmarks.astype(np.float64), m)
    flipped = utils.flip_points(pts, 256, mirror)
    assert np.allclose(utils.flip_points(flipped, 256, mirror), pts)
    # person's right eye (60-67) stays on the left side of the image after flipping
    assert flipped[60:68, 0].mean() < flipped[68:76, 0].mean()


def test_ear_values_are_mirror_symmetric(mirror):
    face = utils.read_faces(DATA, "train")[0]
    pts = face.landmarks.astype(np.float64)
    flipped = utils.flip_points(pts, 2000, mirror)
    assert np.isclose(utils.ear(pts), utils.ear(flipped))


def test_nme_zero_for_perfect_prediction():
    face = utils.read_faces(DATA, "test")[0]
    assert utils.nme(face.landmarks[None], face.landmarks[None])[0] == 0
