#pragma once

#include <iostream>
#include <stdio.h>
#include <string>

#include <haptix.h>
#include <imatrix.h>

#ifdef COIN_TASK
////hyperparameters
constexpr int nq = 40; //todo: correct number?
constexpr int nv = 37;
constexpr int nu = 19;
constexpr int nbody = 38;

bool is_goal_state(mjState*& state, mjBody*& body)
{
	int coins[3] = { mj_name2id("body", "Coin0"), mj_name2id("body", "Coin1"), mj_name2id("body", "Coin2") };

	auto& pos = body->pos;

	//if all are in container, we win
	for (int i = 0; i < 3; ++i)
	{
		int& id = coins[i];
		float& x = pos[id][0];
		float& y = pos[id][1];
		float& z = pos[id][2];

		if (x < .9f && x > .74f && y < -.06f && y > -.22f && z < .1f)
			return true;
	}
	return false; //none matched
}
#endif

#ifdef CARTPOLE_TASK

float reduce_angle(float angle)
{
	if (angle > 3.1415f)
		angle = reduce_angle((angle -= 2 * 3.1415f));
	else if (angle < -3.1415f)
		angle = reduce_angle((angle += 2 * 3.1415f));
	return angle;
}

////hyperparameters
constexpr int nq = 2;
constexpr int nv = 2;
constexpr int nu = 1;
constexpr int nbody = 3;

bool is_goal_state(mjState*& state, mjBody*& body)
{
	int pole_joint_id = mj_name2id("joint", "hinge");

	float theta = 3.141592f - (float)state->qpos[pole_joint_id];

	//if close, true
	if (abs(reduce_angle(theta)) < .1f)
		return true;
	return false;
}
#endif

float clip(float v, float min, float max)
{
	return (v < min ? min : (v > max ? max : v));
}

//for multiple instances, just start multiple windows with the simulation
class SimEnvi
{
public:

	mjState* m_state;
	mjControl* m_control;
	mjBody* m_body;
	mjOneBody* m_bodies[nbody];
	mjInfo* m_info;

	SimEnvi()
	{
		m_state = new mjState();
		m_control = new mjControl();
		m_body = new mjBody();
		m_info = new mjInfo();
		for (int i = 0; i < nbody; ++i)
		{
			m_bodies[i] = new mjOneBody();
			m_bodies[i]->bodyid = i + 1;
		}
		if (mj_connect(NULL) != mjCOM_OK)
			std::cerr << "error loading simulator" << std::endl; //todo: how to connect to different
		get_current_state(m_state, m_bodies); //load in data
		mj_get_control(m_control);
		mj_get_body(m_body);
		mj_info(m_info);
		reset();
	}

	~SimEnvi()
	{
		delete m_state;
		delete m_control;
		delete m_body;
		mj_close(); //todo
	}

	//assumes real control range
	void act(mjControl*&  controls)
	{
		if (mj_set_control(controls) != mjCOM_OK)
			std::cerr << "error setting controls" << std::endl;
		if (mj_step() != mjCOM_OK)
			std::cerr << "error stepping" << std::endl;
	}

	//assumes normalized values
	void act(Matrix2D<float, nu, 1>& controls)
	{
		for (size_t i = 0; i < nu; ++i)
			m_control->ctrl[i] = clip(controls.at(i, 0), -1, 1) * m_info->actuator_ctrlrange[i][1];
		act(m_control);
	}

	void get_current_state(mjState*& state, mjOneBody* bodies[nbody])
	{
		if (mj_get_state(state) != mjCOM_OK)
			std::cerr << "error getting state" << std::endl;
		for (int i = 0; i < nbody; ++i)
			if (mj_get_onebody(bodies[i]) != mjCOM_OK)
				std::cerr << "error getting state" << std::endl;
	}

	void get_current_state(FeatureMap<1, nq + nv, 1>& state)
	{
		get_current_state(m_state, m_bodies);
		for (int i = 0; i < nq; ++i)
		{
			state[0].at(i, 0) = m_state->qpos[i];
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

	float get_current_reward()
	{
		get_current_state(m_state, m_bodies);
		mj_get_body(m_body);

		if (is_goal_state(m_state, m_body))
			return 1;
		else
			return 0;// -.001f; //don't like living todo
	}

	void reset()
	{
		mj_reset(-1);
		get_current_state(m_state, m_bodies); //load in data
		mj_get_control(m_control);
		mj_get_body(m_body);
	}

	void print_body_ids()
	{
		for (int i = 0; i < m_info->nbody; ++i)
		{
			std::cout << i << " is " << mj_id2name("body", i) << std::endl;
		}
	}

	void print_joint_ids()
	{
		for (int i = 0; i < m_info->njnt; ++i)
		{
			std::cout << i << " joint is " << mj_id2name("joint", i) << std::endl;
		}
	}

	void set_current_state(mjState*& state, mjOneBody* bodies[nbody])
	{
		state->nq = m_info->nq;
		state->nv = m_info->nv;
		state->na = m_info->na;
		if (mj_set_state(state) != mjCOM_OK)
			std::cerr << "error setting state" << std::endl;
		/*for (int i = 0; i < nbody; ++i)
			if (mj_set_onebody(bodies[i]) != mjCOM_OK)
				std::cerr << "error setting state" << std::endl;*/
	}

	void save_state(std::string path, mjState*& state, mjOneBody* bodies[nbody])
	{
		FILE* file;
		fopen_s(&file, path.c_str(), "w");
		if (file == NULL)
			std::cerr << "error writing to file" << std::endl;
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
			for (int i = 0; i < nbody; ++i)
			{
				bodies[i]->bodyid = i + 1;
				mj_get_onebody(bodies[i]);
				fwrite(&bodies[i]->pos, sizeof(float), 3, file);
				fwrite(&bodies[i]->quat, sizeof(float), 4, file);
				//fwrite(&bodies[i]->force, sizeof(float), 3, file);
				//fwrite(&bodies[i]->torque, sizeof(float), 3, file);
			}
			fclose(file);
		}
	}
	
	void load_state(std::string path, mjState*& state, mjOneBody* bodies[nbody])
	{
		FILE* file;
		fopen_s(&file, path.c_str(), "r");
		if (file == NULL)
			std::cerr << "error loading from file" << std::endl;
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
			for (int i = 0; i < nbody; ++i)
			{
				bodies[i]->bodyid = i + 1;
				mj_get_onebody(bodies[i]);
				fread(&bodies[i]->pos, sizeof(float), 3, file);
				fread(&bodies[i]->quat, sizeof(float), 4, file);
				//fread(&bodies[i]->force, sizeof(float), 3, file);
				//fread(&bodies[i]->torque, sizeof(float), 3, file);
			}
			fclose(file);
		}
	}
};