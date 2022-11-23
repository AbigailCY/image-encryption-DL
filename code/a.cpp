#include <iostream>
#include <cstdio>
#include <iomanip>
#include <vector>

using namespace std;

int main() {
    int n;
    cin >> hex >> n;
    if (n != 90) {
        return -1;
    }
    cout << hex << setfill('0') << setw(2) << 90;
    while (cin >> hex >> n) {
        int count = 0;
        int prev = 90;
        vector<int> tuple;
        
        while (n != 90) {
            tuple.push_back(n);

            if (n == 91) {
                prev = n;
                cin >> hex >> n;
                if (n == 186) {
                    tuple.push_back(n);
                    prev = 90;

                } else if (n == 187) {
                    tuple.push_back(n);
                    prev = 91;

                } else {
                    if (n == 90) {
                        if (count == prev) {
                            cout << hex << setfill('0') << setw(2) << 90 << ' ';
                            for (int i = 0; i < tuple.size()-1; i++) {
                                cout << hex << setfill('0') << setw(2) << tuple[i] << ' ';
                            }
                        }
                    break;
                    }
                    tuple.push_back(n);
                    count += 1;
                }
            } 
            

            prev = n;
            cin >> hex >> n;
            if (n == 90) {
                if (count == prev) {
                    for (int i = 0; i < tuple.size(); i++) {
                        cout << " " <<  hex << setfill('0') << setw(2) << tuple[i];
                    }
                    cout << ' ' << hex << setfill('0') << setw(2) << 90;
                }
                break;
            }
            count += 1;
        }
    }
    return -1;
}




// 5a 12 5b ba 34 5b bb 88 05 5a 75 cd bb 62 5a 34 cd 78 cc da fb 06 5a
// 5a 12 5b ba 34 5b bb 88 05 5a 34 cd 78 cc da fb 06 5a