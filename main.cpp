#pragma once
#include "stdafx.h"
#include "main.h"
#include "ncon.cpp"

using namespace std;

int main()
{
	cout << "Hello CMake." << endl;
	std::vector<int> v;
	for (int i = 0; i < 2; i++)v.push_back(i);
	std::vector<int> res = vec_where_all(v, 0);
	for (int i = 0; i < (int)res.size(); i++)std::cout << res[i] << ' ';
	std::cout << std::endl;
	return 0;
}

