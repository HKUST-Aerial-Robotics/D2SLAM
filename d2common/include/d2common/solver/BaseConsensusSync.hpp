#pragma once
#include <mutex>
#include <vector>
#include <d2common/utils.hpp>

namespace D2Common {
template <class T>
class BaseSyncDataReceiver {
protected:
    std::recursive_mutex sync_data_recv_lock;
    std::vector<T> sync_datas;
public:
    void add(const T & data) {
        const Utility::Guard lock(sync_data_recv_lock);
        sync_datas.emplace_back(data);
    }
    std::vector<T> retrive(int64_t token, int iteration_count) {
        const Utility::Guard lock(sync_data_recv_lock);
        std::vector<T> datas;
        for (auto it = sync_datas.begin(); it != sync_datas.end(); ) {
            if (it->solver_token == token && it->iteration_count == iteration_count) {
                datas.emplace_back(*it);
                it = sync_datas.erase(it);
            } else {
                it++;
            }
        }
        return datas;
    }
    std::vector<T> retrive_all() {
        const Utility::Guard lock(sync_data_recv_lock);
        std::vector<T> datas = sync_datas;
        sync_datas.clear();
        return datas;
    }
};
}