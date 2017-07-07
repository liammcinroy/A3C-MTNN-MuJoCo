#pragma once

#include <iostream>
#include <stdio.h>
#include <string>

//IF CHANGE VERSION, MUST CHANGE IMPORTED LIBRARIES
#ifndef _DEBUG
#define MUJOCO_PRO
#endif

#ifdef MUJOCO_PRO
#include <mujoco.h>
#include "haptix_wrapper.h"
#else
#include <haptix.h>
#endif

#include <imatrix.h>

#define BALANCE_TASK
//#define REACHER_TASK

#ifdef BALANCE_TASK

//number of checkpoints for the task
#define NUM_DEMONSTRATIONS 50

#define MAX_FRAMES 500

#define MODEL_PATH "C://RL//mjpro131//model//cartpoleBalance.xml"

#define TASK_NAME "balance"

////hyperparameters
//number of generalized positions
constexpr int nq = 2;
//number of generalized velocities
constexpr int nv = 2;
//number of actuators
constexpr int nu = 1;
//number of bodies
constexpr int nbody = 3;
#endif

#ifdef REACHER_TASK

#define FIXED

#define MAX_FRAMES 250

//number of checkpoints for the task
#ifdef FIXED
#define NUM_DEMONSTRATIONS 11
#else
#define NUM_DEMONSTRATIONS 24
#endif 

#define MODEL_PATH "C://RL//mjpro131//model//reacher.xml"

#define TASK_NAME "reacher"

////hyperparameters
//number of generalized positions
constexpr int nq = 4;
//number of generalized velocities
constexpr int nv = 4;
//number of actuators
constexpr int nu = 2;
//number of bodies
constexpr int nbody = 5;
#endif

float reduce_angle(float angle)
{
	if (angle > 3.1415f)
		angle = reduce_angle((angle -= 2 * 3.1415f));
	else if (angle < -3.1415f)
		angle = reduce_angle((angle += 2 * 3.1415f));
	return angle;
}

//for multiple instances, just start multiple windows with the simulation
class SimEnvi
{
#ifdef BALANCE_TASK
private:
	bool gone_up = false;
	
	float reduce_angle(float angle)
	{
		if (angle > 3.1415f)
			angle = reduce_angle((angle -= 2 * 3.1415f));
		else if (angle < -3.1415f)
			angle = reduce_angle((angle += 2 * 3.1415f));
		return angle;
	}

	float get_state_reward(mjState*& state, mjOneBody* bodies[nbody])
	{
#ifdef MUJOCO_PRO
		int pole_joint_id = mj_name2id(model, mjOBJ_JOINT, "hinge");
#else
		int pole_joint_id = mj_name2id("joint", "hinge");
#endif

		float theta = 3.141592f - (float)state->qpos[pole_joint_id];
		float t2 = reduce_angle(theta);

		if (!gone_up && abs(t2) < 3.1415f / 4)
			gone_up = true;
		if ((gone_up && abs(t2) > 3.1415f / 2))
		{
			gone_up = false; //reset state
			terminal = true;
			return -1; //todo: move implementation more towards this
		}
		
		return 0;
	}
#endif

#ifdef REACHER_TASK
private:
	float get_state_reward(mjState*& state, mjOneBody* bodies[nbody])
	{
#ifdef MUJOCO_PRO
		int target_body_id = mj_name2id(model, mjOBJ_BODY, "target"), fingertip_body_id = mj_name2id(model, mjOBJ_BODY, "fingertip");
#else
		int target_body_id = mj_name2id("body", "target"), fingertip_body_id = mj_name2id("body", "fingertip");
#endif

		float distsqrd = 0.0f;
		float velsumsqrd = 0.0f;
		float actsumsqrd = 0.0f;
		for (int i = 0; i < 3; i++)
		{
			distsqrd += pow(bodies[target_body_id]->pos[i] - bodies[fingertip_body_id]->pos[i], 2);
			velsumsqrd += pow(bodies[fingertip_body_id]->linvel[i], 2);
		}

		for (int i = 0; i < nu; ++i)
			actsumsqrd += pow(m_control->ctrl[i], 2);

		float distance = sqrt(distsqrd);

		if (distance < .03f && velsumsqrd < .2f)
		{
			//std::cout << "\tTASK COMPLETED: " << distance << "," << velsumsqrd << std::endl;
			terminal = true;
			return 1;
		}

		else if (n_steps >= MAX_FRAMES)
		{
			terminal = true;
			return 0; //todo:reward may be off?
		}

		//return (-distance + -actsumsqrd) / 100;
		return 0;
	}
#endif

public:

#ifdef MUJOCO_PRO
	//pro
	mjModel* model = NULL;
	mjData* data = NULL;
#endif

	//haptix
	mjState* m_state;
	mjControl* m_control;
	mjBody* m_body;
	mjOneBody* m_bodies[nbody];
	mjInfo* m_info;


	bool terminal = false;
	int n_steps = 0;

	SimEnvi()
	{
		m_state = new mjState();
		m_control = new mjControl();
		m_body = new mjBody();
		m_info = new mjInfo();
		for (int i = 0; i < nbody; ++i)
		{
			m_bodies[i] = new mjOneBody();
			m_bodies[i]->bodyid = i;
		}

#ifdef MUJOCO_PRO

		char error[1000];
		model = mj_loadXML(MODEL_PATH, NULL, error, 1000);
		data = mj_makeData(model);
		mj_resetData(model, data);
#else
		if (mj_connect(NULL) != mjCOM_OK)
			std::cerr << "MuJoCo Haptix: error loading simulator" << std::endl;
		get_current_state(m_state, m_bodies); //load in data
		mj_get_control(m_control);
		mj_get_body(m_body);
		mj_info(m_info);
#endif
	}

	~SimEnvi()
	{
		delete m_state;
		delete m_control;
		delete m_body;
		
#ifdef MUJOCO_PRO
		mj_deleteData(data);
		mj_deleteModel(model);
#else
		mj_close(); //todo
#endif
	}

	//assumes real control range
	void act(mjControl*&  controls)
	{
#ifdef MUJOCO_PRO
		for (size_t i = 0; i < nu; ++i)
		{
			if (controls->ctrl[i] > model->actuator_ctrlrange[i * 2 + 1])
				controls->ctrl[i] = model->actuator_ctrlrange[i * 2 + 1];
			else if (controls->ctrl[i] < model->actuator_ctrlrange[i * 2])
				controls->ctrl[i] = model->actuator_ctrlrange[i * 2];
		}

		for (int i = 0; i < 2; ++i)
		{
			mj_step1(model, data);
			for (size_t i = 0; i < nu; ++i)
				data->ctrl[i] = controls->ctrl[i];
			mj_step2(model, data);
		}
#else
		for (size_t i = 0; i < nu; ++i)
		{
			if (controls->ctrl[i] > m_info->actuator_ctrlrange[i][1])
				controls->ctrl[i] = m_info->actuator_ctrlrange[i][1];
			else if (controls->ctrl[i] < m_info->actuator_ctrlrange[i][0])
				controls->ctrl[i] = m_info->actuator_ctrlrange[i][0];
		}

		if (mj_set_control(controls) != mjCOM_OK)
			std::cerr << "MuJoCo Haptix: error setting controls" << std::endl;
		for (int i = 0; i < 2; ++i)
			if (mj_step() != mjCOM_OK)
				std::cerr << "MuJoCo Haptix: error stepping" << std::endl;
#endif
	}

	//assumes normalized values
	void act(Matrix2D<float, nu, 1>& controls)
	{
#ifdef MUJOCO_PRO
		for (size_t i = 0; i < nu; ++i)
			m_control->ctrl[i] = controls.at(i, 0) * model->actuator_ctrlrange[2 * i + 1];
#else
		for (size_t i = 0; i < nu; ++i)
			m_control->ctrl[i] = controls.at(i, 0) * m_info->actuator_ctrlrange[i][1];
#endif

		act(m_control);
	}

	void get_current_state(mjState*& state, mjOneBody* bodies[nbody])
	{
#ifdef MUJOCO_PRO
		for (size_t i = 0; i < nu; ++i)
			m_state->act[i] = data->act[i];
		for (size_t i = 0; i < nq; ++i)
		{
			m_state->qpos[i] = data->qpos[i];
			m_state->qvel[i] = data->qvel[i];
		}
		for (size_t i = 0; i < nbody; ++i)
		{
			for (size_t k = 0; k < 3; ++k)
			{
				bodies[i]->pos[k] = data->xpos[i * 3 + k];
				bodies[i]->linvel[k] = data->cvel[i * 6 + 3 + k];
				bodies[i]->linacc[k] = data->cacc[i * 6 + 3 + k];
				bodies[i]->quat[k] = data->xquat[i * 4 + k];
			}
			bodies[i]->quat[3] = data->xquat[i * 4 + 3];
		}
#else
		if (mj_get_state(state) != mjCOM_OK)
			std::cerr << "MuJoCo Haptix: error getting state" << std::endl;
		for (int i = 0; i < nbody; ++i)
			if (mj_get_onebody(bodies[i]) != mjCOM_OK)
				std::cerr << "MuJoCo Haptix: error getting onebody" << std::endl;
#endif
	}

	void get_current_state(FeatureMap<1, nq + nv, 1>& state)
	{
		get_current_state(m_state, m_bodies);
		for (int i = 0; i < nq; ++i)
		{
			state[0].at(i, 0) = m_state->qpos[i];

#ifndef USE_PREPROCESSED_INPUT
#ifdef REACHER_TASK
#ifdef MUJOCO_PRO
			int joint0_id = mj_name2id(model, mjOBJ_JOINT, "joint0");
#else
			int joint0_id = mj_name2id("joint", "joint0");
#endif

			if (i == joint0_id)
				state[0].at(i, 0) = reduce_angle(m_state->qpos[i]);
#endif

#ifdef BALANCE_TASK
#ifdef MUJOCO_PRO
			int hinge_id = mj_name2id(model, mjOBJ_JOINT, "hinge");
#else
			int hinge_id = mj_name2id("joint", "hinge");
#endif

			if (i == hinge_id)
				state[0].at(i, 0) = reduce_angle(3.141592f - (float)m_state->qpos[i]);
#endif
#endif

			/*float* mem = &state[0].at(7 * i, 0);
			memcpy(mem, &m_bodies[i]->pos, sizeof(float) * 3);
			memcpy(mem + 3 * sizeof(float), &m_bodies[i]->quat, sizeof(float) * 4);*/

			/*auto& mem = state[0];
			size_t idx = i * 7;
			mem.at(idx + 0, 0) = m_bodies[i]->pos[0];
			mem.at(idx + 1, 0) = m_bodies[i]->pos[2];
			mem.at(idx + 3, 0) = m_bodies[i]->pos[3];
			mem.at(idx + 0, 0) = m_bodies[i]->quat[0];
			mem.at(idx + 1, 0) = m_bodies[i]->quat[2];
			mem.at(idx + 3, 0) = m_bodies[i]->quat[3];
			mem.at(idx + 4, 0) = m_bodies[i]->quat[4];*/
		}

		for (int i = 0; i < nv; ++i)
			state[0].at(nq + i, 0) = m_state->qvel[i];
	}

	//ONLY CALL ONCE PER FRAME!
	float get_current_reward()
	{
		n_steps++;
		get_current_state(m_state, m_bodies);

		return get_state_reward(m_state, m_bodies);
	}

	bool episode_ended()
	{
		if (n_steps >= MAX_FRAMES)
			terminal = true;
		if (terminal)
			return true;
		return false;
	}

	void reset()
	{
#ifdef MUJOCO_PRO
		mj_resetData(model, data);
#else
		mj_reset(-1);
#endif

		//class variables
		terminal = false;
		n_steps = 0;

		get_current_state(m_state, m_bodies);

		//task specific stuff
#ifdef REACHER_TASK
#ifndef FIXED

		int targetx_id = mj_name2id("joint", "target_x"), targety_id = mj_name2id("joint", "target_y");
		float x = ((rand() * 1.0) / RAND_MAX) * .4f - .2f;
		float y = ((rand() * 1.0) / RAND_MAX) * .4f - .2f;
		
		m_state->qpos[targetx_id] = x;
		m_state->qpos[targety_id] = y;

		mj_set_state(m_state);
#endif
#endif
	}

	void print_body_ids()
	{
#ifdef MUJOCO_PRO
		for (int i = 0; i < model->nbody; ++i)
		{
			std::cout << i << " is " << mj_id2name(model, mjOBJ_BODY, i) << std::endl;
		}
#else
		for (int i = 0; i < m_info->nbody; ++i)
		{
			std::cout << i << " is " << mj_id2name("body", i) << std::endl;
		}
#endif
	}

	void print_joint_ids()
	{
#ifdef MUJOCO_PRO
		for (int i = 0; i < model->njnt; ++i)
		{
			std::cout << i << " is " << mj_id2name(model, mjOBJ_JOINT, i) << std::endl;
		}
#else
		for (int i = 0; i < m_info->njnt; ++i)
		{
			std::cout << i << " joint is " << mj_id2name("joint", i) << std::endl;
		}
#endif
	}

	void print_mjinfo()
	{
#ifdef MUJOCO_PRO
		std::cout << "nbody: " << model->nbody << ", nq: " << model->nq << ", nv: " << model->nv << ", nu: " << model->nu << std::endl;
#else
		std::cout << "nbody: " << m_info->nbody << ", nq: " << m_info->nq << ", nv: " << m_info->nv << ", nu: " << m_info->nu << std::endl;
#endif
	}

	void set_current_state(mjState*& state, mjOneBody* bodies[nbody])
	{
#ifdef MUJOCO_PRO
		for (size_t i = 0; i < nu; ++i)
			data->act[i] = state->act[i];
		for (size_t i = 0; i < nq; ++i)
		{
			data->qpos[i] = state->qpos[i];
			data->qvel[i] = state->qvel[i];
		}
		for (size_t i = 0; i < nbody; ++i)
		{
			for (size_t k = 0; k < 3; ++k)
			{
				data->xpos[i * 3 + k] = bodies[i]->pos[k];
				data->cvel[i * 6 + 3 + k] = bodies[i]->linvel[k];
				data->cacc[i * 6 + 3 + k] = bodies[i]->linacc[k];
				data->xquat[i * 4 + k] = bodies[i]->quat[k];
			}
			data->xquat[i * 4 + 3] = bodies[i]->quat[3];
		}
#else
		if (mj_set_state(state) != mjCOM_OK)
			std::cerr << "MuJoCo Haptix: error setting state" << std::endl;
		for (int i = 0; i < nbody; ++i)
			if (mj_set_onebody(bodies[i]) != mjCOM_OK)
				std::cerr << "MuJoCo Haptix: error setting state" << std::endl;
#endif
	}

	void save_state(std::string path, mjState*& state, mjOneBody* bodies[nbody])
	{
		FILE* file;
		fopen_s(&file, path.c_str(), "w");
		if (file == NULL)
			std::cerr << "Error writing to file" << std::endl;
		else
		{
			//begin writing
			char buffer[65536];
			char* pos = &buffer[0];
			for (int i = 0; i < nq; ++i)
				fwrite(&m_state->qpos[i], sizeof(float), 1, file);
			for (int i = 0; i < nv; ++i)
				fwrite(&m_state->qvel[i], sizeof(float), 1, file);
			for (int i = 0; i < nu; ++i)
				fwrite(&m_state->act[i], sizeof(float), 1, file);

			//all bodies
#ifndef MUJOCO_PRO
			for (int i = 0; i < nbody; ++i)
			{
				bodies[i]->bodyid = i;
				mj_get_onebody(bodies[i]);
				fwrite(&bodies[i]->pos, sizeof(float), 3, file);
				fwrite(&bodies[i]->quat, sizeof(float), 4, file);
				//fwrite(&bodies[i]->force, sizeof(float), 3, file);
				//fwrite(&bodies[i]->torque, sizeof(float), 3, file);
			}
#endif
			fclose(file);
		}
	}

	void load_state(std::string path, mjState*& state, mjOneBody* bodies[nbody])
	{
		FILE* file;
		fopen_s(&file, path.c_str(), "r");
		if (file == NULL)
			std::cerr << "Error loading from file" << std::endl;
		else
		{
			//begin reading
			for (int i = 0; i < nq; ++i)
				fread(&state->qpos[i], sizeof(float), 1, file);
			for (int i = 0; i < nv; ++i)
				fread(&state->qvel[i], sizeof(float), 1, file);
			for (int i = 0; i < nu; ++i)
				fread(&state->act[i], sizeof(float), 1, file);

			//all bodies
#ifdef MUJOCO_PRO
			for (int i = 0; i < nbody; ++i)
			{
				fread(&data->xpos[i * 3], sizeof(float), 3, file);
				fread(&data->xquat[i * 4], sizeof(float), 4, file);
			}
#else
			for (int i = 0; i < nbody; ++i)
			{
				bodies[i]->bodyid = i;
				mj_get_onebody(bodies[i]);
				fread(&bodies[i]->pos, sizeof(float), 3, file);
				fread(&bodies[i]->quat, sizeof(float), 4, file);
				//fread(&bodies[i]->force, sizeof(float), 3, file);
				//fread(&bodies[i]->torque, sizeof(float), 3, file);
			}
#endif
			fclose(file);
		}
	}

};