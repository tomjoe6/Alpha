#include <iostream>
#include <vector>
using namespace std;

bool check(const vector<int>& nums, int m, int mid) {
    int cnt = 1;
    int sum = 0;
    for (int num : nums) {
        if (sum + num > mid) {
            cnt++;
            sum = num;
        } else {
            sum += num;
        }
    }
    return cnt <= m;
}

int main() {
    int n, m;
    cin >> n >> m;
    vector<int> nums(n);
    int low = 0, high = 0;
    for (int i = 0; i < n; i++) {
        cin >> nums[i];
        low = max(low, nums[i]);
        high += nums[i];
    }
    while (low < high) {
        int mid = (low + high) / 2;
        if (check(nums, m, mid)) {
            high = mid;
        } else {
            low = mid + 1;
        }
    }
    cout << low << endl;
    return 0;
}