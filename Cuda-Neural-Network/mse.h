#pragma once

#include "device_launch_parameters.h"

__device__ float getMSE(float target, float prediction) {
	return 2 * (prediction - target);
}
