#include <d2common/utils.hpp>

using namespace D2Common;

void testQuaternionAveraging() {
  std::vector<Eigen::Quaterniond> qs{Eigen::Quaterniond(1.0, -0.2, 0.0, 0.0),
                                     Eigen::Quaterniond(1.0, 0.4, 0.0, 0.0),
                                     Eigen::Quaterniond(-1.0, 0.0, 0.0, 0.0)};
  for (auto q : qs) {
    q.normalize();
  }
  auto q = Utility::averageQuaterions<double>(qs);
  std::cout << "q.w() " << q.w() << " xyz " << q.vec().transpose() << std::endl;
}

int main() { testQuaternionAveraging(); }