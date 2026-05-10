import unittest

from analysis.event_detector import detect_shots_from_tracks


FPS = 30.0
PX_PER_METER = 100.0
PLAYER = (80.0, 40.0, 150.0, 190.0)


def player_frames(count):
    return [PLAYER for _ in range(count)]


def moving_player_frames(count, start_x, step_x):
    frames = []
    for frame in range(count):
        x1 = start_x + step_x * frame
        frames.append((x1, 40.0, x1 + 70.0, 190.0))
    return frames


def empty_track(count):
    return [None for _ in range(count)]


class EventDetectorTests(unittest.TestCase):
    def test_accepts_true_shot_from_contact_and_travel(self):
        traj = empty_track(32)
        for frame in range(0, 11):
            traj[frame] = (120.0, 180.0)
        for frame in range(11, 24):
            traj[frame] = (120.0 - 30.0 * (frame - 10), 180.0)

        shots, candidates = detect_shots_from_tracks(
            {1: traj}, FPS, PX_PER_METER, player_frames(32)
        )

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].track_id, 1)
        self.assertTrue(any(c.accepted for c in candidates))

    def test_rejects_fast_touch_without_post_contact_travel(self):
        traj = empty_track(24)
        for frame in range(0, 11):
            traj[frame] = (120.0, 180.0)
        for frame, x in [(11, 102.0), (12, 84.0), (13, 66.0)]:
            traj[frame] = (x, 180.0)
        for frame in range(14, 24):
            traj[frame] = (66.0, 180.0)

        shots, candidates = detect_shots_from_tracks(
            {1: traj}, FPS, PX_PER_METER, player_frames(24)
        )

        self.assertEqual(len(shots), 0)
        self.assertTrue(any(
            c.rejection_reason in {
                "insufficient_post_contact_travel",
                "ball_stayed_with_player",
                "not_goalward",
            }
            for c in candidates
        ))

    def test_detects_variable_number_of_shots(self):
        first = empty_track(80)
        second = empty_track(80)
        for frame in range(0, 11):
            first[frame] = (120.0, 180.0)
        for frame in range(11, 24):
            first[frame] = (120.0 - 30.0 * (frame - 10), 180.0)
        for frame in range(40, 51):
            second[frame] = (120.0, 180.0)
        for frame in range(51, 64):
            second[frame] = (120.0 - 28.0 * (frame - 50), 180.0)

        shots, _ = detect_shots_from_tracks(
            {1: first, 2: second}, FPS, PX_PER_METER, player_frames(80)
        )

        self.assertEqual(len(shots), 2)

    def test_no_shot_video_returns_empty(self):
        traj = [(120.0, 180.0) for _ in range(30)]

        shots, candidates = detect_shots_from_tracks(
            {1: traj}, FPS, PX_PER_METER, player_frames(30)
        )

        self.assertEqual(shots, [])
        self.assertEqual(candidates, [])

    def test_global_candidate_handles_track_id_switch(self):
        before = empty_track(30)
        after = empty_track(30)
        for frame in range(0, 11):
            before[frame] = (120.0, 180.0)
        for frame in range(11, 24):
            after[frame] = (20.0 - 30.0 * (frame - 11), 180.0)

        shots, candidates = detect_shots_from_tracks(
            {1: before, 9: after}, FPS, PX_PER_METER, player_frames(30)
        )

        self.assertEqual(len(shots), 1)
        self.assertEqual(shots[0].source, "global")
        self.assertTrue(any(c.accepted and c.source == "global" for c in candidates))

    def test_rejects_goalward_ball_handling_that_moves_with_player(self):
        traj = empty_track(36)
        for frame in range(8, 28):
            traj[frame] = (190.0 - 10.0 * (frame - 8), 180.0)

        shots, candidates = detect_shots_from_tracks(
            {1: traj},
            FPS,
            PX_PER_METER,
            moving_player_frames(36, 145.0, -10.0),
        )

        self.assertEqual(len(shots), 0)
        self.assertTrue(any(c.rejection_reason == "ball_stayed_with_player" for c in candidates))

    def test_rejects_fast_motion_away_from_goal(self):
        traj = empty_track(32)
        for frame in range(0, 11):
            traj[frame] = (120.0, 180.0)
        for frame in range(11, 24):
            traj[frame] = (120.0 + 30.0 * (frame - 10), 180.0)

        shots, candidates = detect_shots_from_tracks(
            {1: traj}, FPS, PX_PER_METER, player_frames(32)
        )

        self.assertEqual(len(shots), 0)
        self.assertTrue(any(c.rejection_reason == "not_goalward" for c in candidates))

    def test_rejects_unrealistic_velocity_jump(self):
        traj = empty_track(16)
        for frame in range(0, 11):
            traj[frame] = (120.0, 180.0)
        traj[11] = (-300.0, -220.0)

        shots, candidates = detect_shots_from_tracks(
            {1: traj}, FPS, PX_PER_METER, player_frames(16)
        )

        self.assertEqual(len(shots), 0)
        self.assertTrue(any(c.rejection_reason == "unrealistic_velocity" for c in candidates))


if __name__ == "__main__":
    unittest.main()
