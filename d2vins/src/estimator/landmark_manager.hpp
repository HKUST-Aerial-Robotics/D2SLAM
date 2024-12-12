#pragma once

#include <d2common/d2vinsframe.h>

#include "d2frontend/d2landmark_manager.h"

namespace D2VINS {
class D2EstimatorState;
class D2LandmarkManager : public D2FrontEnd::LandmarkManager {
    std::map<LandmarkIdType, StatePtr> landmark_state;
    int estimated_landmark_size = 0;
    void initialLandmarkState(LandmarkPerId &lm, const D2EstimatorState *state);

  public:
    virtual void addKeyframe(const VisualImageDescArray &images, double td);
    std::vector<LandmarkPerId>
    availableMeasurements(int max_pts, int max_solve_measurements,
                          const std::set<FrameIdType> &current_frames) const;
    StatePtr getLandmarkState(LandmarkIdType landmark_id) const;
    void initialLandmarks(const D2EstimatorState *state);
    void syncState(const D2EstimatorState *state);
    int outlierRejection(const D2EstimatorState *state,
                          const std::set<LandmarkIdType> &used_landmarks);
    int outlierRejectionByScale(const D2EstimatorState *state,
                          const std::set<LandmarkIdType> &used_landmarks);
    void moveByPose(const Swarm::Pose &delta_pose);
    virtual void removeLandmark(const LandmarkIdType &id) override;
    std::map<FrameIdType, Swarm::Pose>
    SFMInitialization(const std::vector<VINSFramePtr>& frames, int camera_idx);
    std::map<LandmarkIdType, Vector3d>
    triangulationFrames(FrameIdType frame1_id, const Swarm::Pose &frame1,
                        FrameIdType frame2_id, const Swarm::Pose &frame2,
                        int camera_idx);
    std::map<LandmarkIdType, Vector3d>
    triangulationFrames(const std::map<FrameIdType, Swarm::Pose> &frame_poses,
                        int camera_idx, int min_tracks);
    bool SolveRelativePose5Pts(Swarm::Pose &ret, int camera_idx,
                               FrameIdType frame1_id, FrameIdType frame2_id);
    bool InitFramePoseWithPts(
        Swarm::Pose &ret,
        std::map<LandmarkIdType, Vector3d> &last_triangluation_pts,
        FrameIdType frame_id, int camera_idx);
    const std::map<FrameIdType, Swarm::Pose>
    PerformBA(const std::map<FrameIdType, Swarm::Pose> &initial,
              const VINSFramePtr& last_frame, const VINSFramePtr& head_frame_for_match,
              std::map<LandmarkIdType, Vector3d> initial_pts,
              int camera_idx) const;
};

} // namespace D2VINS