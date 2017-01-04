#include <iostream>
#include <time.h>
#include <tuple>
#include <Windows.h>

#include <imatrix.h>
#include <ilayer.h>
#include <neuralnet.h>
#include <neuralnetanalyzer.h>

#include "mjcCartpole.h"
#include "a3c.h"

using namespace std;

typedef NeuralNetAnalyzer<GlobalActor> Analyzer;

#define TRAINING true

#define NUM_DEMONSTRATIONS 50

int main(int argc, char* argv[])
{
	//start both with the same parameters
	auto actor_default_path = CSTRING("actor_default.nn");
	auto critic_default_path = CSTRING("critic_default.nn");

	using actor_default_t = decltype(actor_default_path);
	using critic_default_t = decltype(critic_default_path);

	GlobalActor::save_data<actor_default_t>();
	GlobalCritic::save_data<critic_default_t>();
#define USE_DEMONSTRATIONS
	{
		srand(0);

		//this is for saving the average score to output every 20 training episodes
		Analyzer::sample_size = 50;

		float start_t = 0;
		float last_epoch_t = 0;

		actor_threads = vector<GlobalActor>(NUM_THREADS);
		critic_threads = vector<GlobalCritic>(NUM_THREADS);
#ifdef USE_DEMONSTRATIONS
		auto actor_save_path = CSTRING("actor_dem.nn");
		auto critic_save_path = CSTRING("critic_dem.nn");
#else
		auto actor_save_path = CSTRING("actor_nodem.nn");
		auto critic_save_path = CSTRING("critic_nodem.nn");
#endif

		using actor_save_t = decltype(actor_save_path);
		using critic_save_t = decltype(critic_save_path);

		//start environment
		CartpoleSimEnvi sim{};

		if (!TRAINING)
		{
			GlobalActor::load_data<actor_save_t>();
			GlobalCritic::load_data<critic_save_t>();
		}

		size_t T = 0;
		size_t ep = 0;
		int num_epoch = -1;
		start_t = clock();
		while (true && ep < 3000) // stop after 2000 episodes
		{
			//divy up samples
			get_samples();

#pragma omp parallel num_threads(NUM_THREADS) reduction(+:T) shared(actor_threads, critic_threads)
			for (int i = 0; i < NUM_THREADS && ep > 0 && TRAINING; ++i)
			{
				mj_reset(-1);

#ifdef USE_DEMONSTRATIONS
				int num = rand() % NUM_DEMONSTRATIONS;
				sim.load_state("demonstrations//save" + to_string(num) + ".dat", sim.m_state, sim.m_bodies);
				sim.set_current_state(sim.m_state, sim.m_bodies);
#endif

				//run a3c
				float score = A3C(T, i /*omp_get_thread_num()*/, sim);

#pragma omp critical
				//cout << score << endl;
			}

			cout << ep << endl;

			//update weights
			//GlobalActor::apply_gradient();
			//GlobalCritic::apply_gradient();

#ifdef FIXED_VARIANCE
			if (variance > .005f)
				variance *= .999f;
#endif

			//anneal weights
			if (ep % 10 == 0)
			{
				GlobalActor::learning_rate *= .99f;
				GlobalCritic::learning_rate *= .99f;
			}

			if (ep % 20 == 0)
			{
				float avg = 0.0f;
				cout << (clock() - start_t) / CLOCKS_PER_SEC << " sec" << endl;
				cout << "TESTING...";
				cout.flush();
				Sleep(500);

				InputFM input{ 0 };

				int cart_id = mj_name2id("body", "cart"), cart_joint_id = mj_name2id("joint", "slider"), pole_id = mj_name2id("site", "tip"), pole_joint_id = mj_name2id("joint", "hinge");
				mjState* currentState = new mjState();
				mjOneBody* cart = new mjOneBody();
				mjControl* control = new mjControl();
				cart->bodyid = cart_id;
				for (size_t tests = 0; tests < Analyzer::sample_size; ++tests)
				{
					mj_reset(-1);

					int i = 0;

					bool gone_up = false;

#ifdef BALANCE
					gone_up = true;
#endif

					//run a test
					bool terminal = false;
					float score = 0;
					float act = 0;
					size_t frame = 0;
					InputFM input{ 0 };

					cart->bodyid = cart_id;
					mj_get_state(currentState);
					mj_get_onebody(cart);
					mj_get_control(control);

					////training step
					while (!terminal)
					{
						float reward = 0.0f;
						float theta = 3.141592f - (float)currentState->qpos[pole_joint_id];
						for (size_t t = 0; !terminal; ++t)
						{
							float t2 = reduce_angle(theta);
							if (!gone_up && abs(t2) < 3.1415f / 4)
								gone_up = true;
							if ((gone_up && abs(t2) > 3.1415f / 2))
								terminal = true;
#ifdef BALANCE
							if (terminal)
								reward = -1;
#else
							if (terminal)
							{
								if (gone_up && abs(t2) < 3.1415f / 2)
									reward = 0;// 1;
								else if (gone_up) //-1 * .99^500
									reward = -10 * .006;// (tMAX - frame) * REWARD_FUNC(3.1415f / 2);
								else
									reward = -10;
							}
#endif

							if (frame >= tMAX - 1)
								terminal = true;

							score += reward;
							if (t >= tMAX || terminal)
								break;
							mj_get_state(currentState);
							mj_get_onebody(cart);
							mj_get_control(control);
							theta = 3.141592f - (float)currentState->qpos[pole_joint_id]; //reduce_angle();
							input = {};
							input[0].at(0, 0) = theta;
							input[0].at(1, 0) = (float)currentState->qvel[pole_joint_id];
							input[0].at(2, 0) = (float)cart->pos[0];
							input[0].at(3, 0) = (float)cart->linvel[0];
							//sim.get_current_state(input);

							//get next action
							GlobalActor::discriminate(input);

							//get max act
							act = get_action(GlobalActor::template get_batch_activations<GlobalActor::last_layer_index>()[0]);

							//pass to environment and get reward
							control->ctrl[0] = act;
							mj_set_control(control);

							for (int i = 0; i < 3; ++i)
								mj_step();

							//calculate reward (get new d->xpos etc.)
							t2 = reduce_angle(theta);
#ifdef BALANCE
							reward = 0;
#else
							//reward = REWARD_FUNC(t2);
							reward = 0;
#endif			
							++frame;

#ifdef USE_DEMONSTRATIONS
							if (!TRAINING && t % 10 == 0)
							{
								sim.get_current_state(sim.m_state, sim.m_bodies);
								sim.save_state("demonstrations//save" + to_string(i) + ".dat", sim.m_state, sim.m_bodies);
								++i;
							}
#endif

						}
					}
					//get discounted reward
					score *= pow(DISCOUNT_FACTOR, frame);

					//add to totals
					Analyzer::add_point(score);
				}
				//mean
				avg = Analyzer::mean_error();
				//quickly get sample variance
				for (size_t i = 0; i < Analyzer::sample_size; ++i)
					Analyzer::add_point(pow(Analyzer::sample[0] - avg, 2) / (Analyzer::sample_size - 1) * Analyzer::sample_size);
				float var = Analyzer::mean_error();

				cout << avg << ',' << var << endl;
				cout << MEAN_TRANSFORM(GlobalActor::template get_batch_activations<GlobalActor::last_layer_index>()[0][0].at(0, 0)) << ',' << VAR_TRANSFORM(GlobalActor::template get_batch_activations<GlobalActor::last_layer_index>()[0][0].at(1, 0)) << endl;


				//save score and variance
#ifdef USE_DEMONSTRATIONS
				Analyzer::save_mean_error("scores_training_dem.dat");
#else
				Analyzer::save_mean_error("scores_training_nodem.dat");
#endif

				delete currentState;
				delete cart;
				delete control;

				GlobalActor::save_data<actor_save_t>();
				GlobalCritic::save_data<critic_save_t>();
			}

			++ep;
		}
	}

	//go to demonstrations
	
	//reload default params
	GlobalActor::load_data<actor_default_t>();
	GlobalCritic::load_data<critic_default_t>();

	GlobalActor::learning_rate = .001f;
	GlobalCritic::learning_rate = .001f;

	//get rid of previous analyzer params
	Analyzer::errors.clear();

	return 0; //todo demonstrations for full cartpole

#ifndef USE_DEMONSTRATIONS
#define USE_DEMONSTRATIONS
#else
#undef USE_DEMONSTRATIONS
#endif

	{
		srand(0);

		//this is for saving the average score to output every 20 training episodes
		Analyzer::sample_size = 50;

		float start_t = 0;
		float last_epoch_t = 0;

		actor_threads = vector<GlobalActor>(NUM_THREADS);
		critic_threads = vector<GlobalCritic>(NUM_THREADS);
#ifdef USE_DEMONSTRATIONS
		auto actor_save_path = CSTRING("actor_dem.nn");
		auto critic_save_path = CSTRING("critic_dem.nn");
#else
		auto actor_save_path = CSTRING("actor_nodem.nn");
		auto critic_save_path = CSTRING("critic_nodem.nn");
#endif

		using actor_save_t = decltype(actor_save_path);
		using critic_save_t = decltype(critic_save_path);

		//start environment
		CartpoleSimEnvi sim{};

		if (!TRAINING)
		{
			GlobalActor::load_data<actor_save_t>();
			GlobalCritic::load_data<critic_save_t>();
		}

		size_t T = 0;
		size_t ep = 0;
		int num_epoch = -1;
		start_t = clock();
		while (true && ep < 3000) // stop after 2000 episodes
		{
			//divy up samples
			get_samples();

#pragma omp parallel num_threads(NUM_THREADS) reduction(+:T) shared(actor_threads, critic_threads)
			for (int i = 0; i < NUM_THREADS && ep > 0 && TRAINING; ++i)
			{
				mj_reset(-1);

#ifdef USE_DEMONSTRATIONS
				int num = rand() % NUM_DEMONSTRATIONS;
				sim.load_state("demonstrations//save" + to_string(num) + ".dat", sim.m_state, sim.m_bodies);
				sim.set_current_state(sim.m_state, sim.m_bodies);
#endif

				//run a3c
				float score = A3C(T, i /*omp_get_thread_num()*/, sim);

#pragma omp critical
				//cout << score << endl;
			}

			cout << ep << endl;

			//update weights
			//GlobalActor::apply_gradient();
			//GlobalCritic::apply_gradient();

#ifdef FIXED_VARIANCE
			if (variance > .005f)
				variance *= .999f;
#endif

			//anneal weights
			if (ep % 10 == 0)
			{
				GlobalActor::learning_rate *= .99f;
				GlobalCritic::learning_rate *= .99f;
			}

			if (ep % 20 == 0)
			{
				float avg = 0.0f;
				cout << (clock() - start_t) / CLOCKS_PER_SEC << " sec" << endl;
				cout << "TESTING...";
				cout.flush();
				Sleep(500);

				InputFM input{ 0 };

				int cart_id = mj_name2id("body", "cart"), cart_joint_id = mj_name2id("joint", "slider"), pole_id = mj_name2id("site", "tip"), pole_joint_id = mj_name2id("joint", "hinge");
				mjState* currentState = new mjState();
				mjOneBody* cart = new mjOneBody();
				mjControl* control = new mjControl();
				cart->bodyid = cart_id;
				for (size_t tests = 0; tests < Analyzer::sample_size; ++tests)
				{
					mj_reset(-1);

					bool gone_up = false;

#ifdef BALANCE
					gone_up = true;
#endif

					//run a test
					bool terminal = false;
					float score = 0;
					float act = 0;
					size_t frame = 0;
					InputFM input{ 0 };

					cart->bodyid = cart_id;
					mj_get_state(currentState);
					mj_get_onebody(cart);
					mj_get_control(control);

					////training step
					while (!terminal)
					{
						float reward = 0.0f;
						float theta = 3.141592f - (float)currentState->qpos[pole_joint_id];
						for (size_t t = 0; !terminal; ++t)
						{
							float t2 = reduce_angle(theta);
							if (!gone_up && abs(t2) < 3.1415f / 4)
								gone_up = true;
							if ((gone_up && abs(t2) > 3.1415f / 2))
								terminal = true;
#ifdef BALANCE
							if (terminal)
								reward = -1;
#else
							if (terminal)
							{
								if (gone_up && abs(t2) < 3.1415f / 2)
									reward = 0;// 1;
								else if (gone_up)
									reward = 0;// (tMAX - frame) * REWARD_FUNC(3.1415f / 2);
								else
									reward = 0;
							}
#endif

							if (frame >= tMAX - 1)
								terminal = true;

							score += reward;
							if (t >= tMAX || terminal)
								break;
							mj_get_state(currentState);
							mj_get_onebody(cart);
							mj_get_control(control);
							theta = 3.141592f - (float)currentState->qpos[pole_joint_id]; //reduce_angle();
							input = {};
							input[0].at(0, 0) = theta;
							input[0].at(1, 0) = (float)currentState->qvel[pole_joint_id];
							input[0].at(2, 0) = (float)cart->pos[0];
							input[0].at(3, 0) = (float)cart->linvel[0];
							//sim.get_current_state(input);

							//get next action
							GlobalActor::discriminate(input);

							//get max act
							act = get_action(GlobalActor::template get_batch_activations<GlobalActor::last_layer_index>()[0]);

							//pass to environment and get reward
							control->ctrl[0] = act;
							mj_set_control(control);

							for (int i = 0; i < 3; ++i)
								mj_step();

							//calculate reward (get new d->xpos etc.)
							t2 = reduce_angle(theta);
#ifdef BALANCE
							reward = 0;
#else
							reward = REWARD_FUNC(t2);
#endif			
							++frame;
						}
					}
					//get discounted reward
					score *= pow(DISCOUNT_FACTOR, frame);

					//add to totals
					Analyzer::add_point(score);
				}
				//mean
				avg = Analyzer::mean_error();
				//quickly get sample variance
				for (size_t i = 0; i < Analyzer::sample_size; ++i)
					Analyzer::add_point(pow(Analyzer::sample[0] - avg, 2) / (Analyzer::sample_size - 1) * Analyzer::sample_size);
				float var = Analyzer::mean_error();

				cout << avg << ',' << var << endl;
				cout << MEAN_TRANSFORM(GlobalActor::template get_batch_activations<GlobalActor::last_layer_index>()[0][0].at(0, 0)) << ',' << VAR_TRANSFORM(GlobalActor::template get_batch_activations<GlobalActor::last_layer_index>()[0][0].at(1, 0)) << endl;

				//save score and variance
#ifdef USE_DEMONSTRATIONS
				Analyzer::save_mean_error("scores_training_dem.dat");
#else
				Analyzer::save_mean_error("scores_training_nodem.dat");
#endif

				delete currentState;
				delete cart;
				delete control;

				GlobalActor::save_data<actor_save_t>();
				GlobalCritic::save_data<critic_save_t>();
			}

			++ep;
		}
	}

	return 0;
}
