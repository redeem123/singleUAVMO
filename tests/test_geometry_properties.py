from hypothesis import given, strategies as st
import numpy as np

from uav_benchmark.core.evaluate_path import _dist_points_to_segments_2d


@given(
    cx=st.floats(min_value=-1000, max_value=1000),
    cy=st.floats(min_value=-1000, max_value=1000),
    sx0=st.floats(min_value=-1000, max_value=1000),
    sy0=st.floats(min_value=-1000, max_value=1000),
    sx1=st.floats(min_value=-1000, max_value=1000),
    sy1=st.floats(min_value=-1000, max_value=1000),
)
def test_dist_points_to_segments_2d_properties(cx, cy, sx0, sy0, sx1, sy1):
    centers = np.array([[cx, cy]])
    seg_starts = np.array([[sx0, sy0]])
    seg_ends = np.array([[sx1, sy1]])
    
    # Avoid zero-length segments for stability in this test
    if np.allclose(seg_starts, seg_ends):
        return
        
    dist = _dist_points_to_segments_2d(centers, seg_starts, seg_ends)
    
    assert dist.shape == (1, 1)
    assert dist[0, 0] >= 0.0
    assert not np.isnan(dist[0, 0])
