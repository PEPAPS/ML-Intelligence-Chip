#include "Pattern.H"

// compare function MUST be here or above usage
bool compare(const pair<double, int>& a, const pair<double, int>& b) {
    return a.first > b.first;
}

// constructor
Pattern::Pattern(sc_module_name name, string img_name)
    : sc_module(name), img_name(img_name)
{
    SC_METHOD(run);
    sensitive << clock.neg();
    cycle = 0;
}

// run function
void Pattern::run()
{
    if (cycle == 0)
    {	
        rst.write(1);
        in_valid.write(0);
    }
    else if (cycle == 1)
    {
        rst.write(0);

        ifstream inputFile(("data/" + img_name).c_str());

        double img_element;
        int cnt{0};

        while (inputFile >> img_element)
        {
            img[cnt] = img_element;
            cnt++;
        }

        in_valid.write(1);
    }
    else if (out_valid.read() == 1)
    {
        ifstream class_file("data/imagenet_classes.txt");

        vector<string> class_name;
        string line;

        while (getline(class_file, line))
            class_name.push_back(line);

        vector<pair<double, int>> indexed_values;

        for (int i = 0; i < 1000; ++i)
            indexed_values.push_back(make_pair(output_softmax[i].read(), i));

        sort(indexed_values.begin(), indexed_values.end(), compare);

        cout << "Top 100 classes:" << endl;
        cout << "=================================================" << endl;
        cout << fixed << setprecision(2);

        for (int i = 0; i < 100; ++i)
        {
            cout << indexed_values[i].second
                 << " | " << output_linear[indexed_values[i].second].read()
                 << " | " << indexed_values[i].first * 100
                 << " | " << class_name[indexed_values[i].second] << endl;
        }

        cout << "=================================================" << endl;
        sc_stop();
    }
    else
    {
        in_valid.write(0);
    }

    cycle++;
    if (cycle == CYCLE)
        exit(0);
}