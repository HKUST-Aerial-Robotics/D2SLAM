#include "landmark_manager.hpp"
#include <d2vins/d2vins_params.hpp>
#include <d2vins/d2vins_types.hpp>

using namespace Eigen;
namespace D2VINS {

class D2EstimatorState {
protected:
    std::vector<VINSFrame*> sld_win;
    std::map<FrameIdType, VINSFrame*> frame_db;
    std::vector<Swarm::Pose> extrinsic; //extrinsic of cameras
    D2LandmarkManager lmanager;
    std::map<FrameIdType, state_type*> _frame_pose_state;
    std::map<FrameIdType, state_type*> _frame_spd_Bias_state;
    std::vector<state_type*> _camera_extrinsic_state;

    void popFrame(int index) {
        //Remove from sliding window
        auto frame_id = sld_win[index]->frame_id;
        if (params->verbose) {
            printf("[D2VSIN::D2EstimatorState] remove frame %ld\n", frame_id);
        }
        delete sld_win[index];
        sld_win.erase(sld_win.begin() + index);
        lmanager.popFrame(frame_id);
        frame_db.erase(frame_id);
        delete _frame_pose_state[frame_id];
        delete _frame_spd_Bias_state[frame_id];
        _frame_pose_state.erase(frame_id);
        _frame_spd_Bias_state.erase(frame_id);
    }

public:
    state_type td = 0.0;
    D2EstimatorState() {}
    void init(std::vector<Swarm::Pose> _extrinsic, double _td) {
        extrinsic = _extrinsic;
        for (auto & pose : extrinsic) {
            auto _p = new state_type[POSE_SIZE];
            pose.to_vector(_p);
            _camera_extrinsic_state.push_back(_p);
        }
        td = _td;
    }

    size_t size() const {
        return sld_win.size();
    }

    VINSFrame & getFrame(int index) {
        return *sld_win[index];
    }

    VINSFrame & firstFrame() {
        return *sld_win[0];
    }

    double * getPoseState(FrameIdType frame_id) const {
        return _frame_pose_state.at(frame_id);
    }

    double * getExtrinsicState(int i) const {
        return _camera_extrinsic_state[i];
    }

    double * getSpdBiasState(FrameIdType frame_id) const {
        return _frame_spd_Bias_state.at(frame_id);
    }
    
    double * getLandmarkState(LandmarkIdType landmark_id) const {
        return lmanager.getLandmarkState(landmark_id);
    }

    std::vector<LandmarkPerId> availableLandmarkMeasurements() const {
        return lmanager.availableMeasurements();
    }

    void clearFrame() {
        if (sld_win.size() >= 2 && !sld_win[sld_win.size() - 1]->is_keyframe) {
            //If last frame is not keyframe then remove it.
            popFrame(sld_win.size() - 1);
        }

        if (sld_win.size() >= params->max_sld_win_size) {
            if (sld_win[sld_win.size() - 2]->is_keyframe) {
                popFrame(0);
            }
        }
    }

    void addFrame(const VisualImageDescArray & images, const VINSFrame & _frame, bool is_keyframe) {
        auto * frame = new VINSFrame;
        *frame = _frame;
        sld_win.push_back(frame);
        frame_db[frame->frame_id] = frame;
        _frame_pose_state[frame->frame_id] = new state_type[POSE_SIZE];
        _frame_spd_Bias_state[frame->frame_id] = new state_type[FRAME_SPDBIAS_SIZE];
        frame->toVector(_frame_pose_state[frame->frame_id], _frame_spd_Bias_state[frame->frame_id]);

        lmanager.addKeyframe(images, td);
        if (params->verbose) {
            printf("[D2VINS::D2EstimatorState] add frame %ld, current %ld frame\n", images.frame_id, sld_win.size());
        }
    }

    void syncFromState() {
        //copy state buffer to structs.
        //First sync the poses
        for (auto it : _frame_pose_state) {
            auto frame_id = it.first;
            frame_db.at(frame_id)->fromVector(it.second, _frame_spd_Bias_state.at(frame_id));
        }

        for (size_t i = 0; i < extrinsic.size(); i ++ ) {
            extrinsic[i].from_vector(_camera_extrinsic_state[i]);
        }

        lmanager.syncState(extrinsic, frame_db);
    }

    void pre_solve() {
        lmanager.initialLandmarks(frame_db, extrinsic);
    }

    VINSFrame lastFrame() const {
        assert(sld_win.size() > 0 && "SLDWIN size must > 1 to call D2EstimatorState::lastFrame()");
        return *sld_win.back();
    }

    std::vector<LandmarkPerId> getInitializedLandmarks() const {
        return lmanager.getInitializedLandmarks();
    }

};
}