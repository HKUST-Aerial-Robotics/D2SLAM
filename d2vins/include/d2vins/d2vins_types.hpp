#include "d2frontend/d2frontend_types.h"

namespace D2VINS {
    struct IMUData {
        double t = 0.0;
        double dt = 0.0;
        Vector3d acc;
        Vector3d gyro;
        IMUData(): acc(0.0, 0.0, 0.0),gyro(0.0, 0.0, 0.0){}
    };

    struct IMUBuffer
    {
        std::vector<IMUData> buf;
        double t_last = 0.0;
        void add(const IMUData & data) {
            buf.emplace_back(data);
            t_last = data.t;
        }

        Vector3d mean_acc() const {
            Vector3d acc_sum(0, 0, 0);
            for (auto & data : buf) {
                acc_sum += data.acc;
            }
            return acc_sum/size();
        }

        size_t size() const {
            return buf.size();
        }

        bool avaiable(double t) const {
            return t_last > t;
        }
    };

    
}