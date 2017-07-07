#include <array>
#include <iostream>
#include <time.h>
#include <tuple>
#include <Windows.h>

#include <imatrix.h>
#include <ilayer.h>
#include <neuralnet.h>
#include <neuralnetanalyzer.h>

////Do macros here to carry into other files

#define TRAINING true

//generic a3c or hcr
#define USE_DEMONSTRATIONS

//random sampling or greedy
//#define USE_RANDOM_SAMPLING

//use raw input (to test power of algorithm to learn relevant features)
#define USE_PREPROCESSED_INPUTS

#define NUM_TRAINING_EPISODES 1500
#define EPISODES_BETWEEN_TESTS 20
#define SAVE_DATA

#include "mjcEnvironment.h"
#include "a3c.h"

using namespace std;

typedef NeuralNetAnalyzer<GlobalActor> Analyzer;

int main(int argc, char* argv[])
{
#ifdef MUJOCO_PRO
	mj_activate("C://RL//mjpro131//bin//mjkeyOLD.txt");
#endif

	float start_t = 0;
	float last_epoch_t = 0;

	actor_threads = vector<GlobalActor>(NUM_THREADS);
	critic_threads = vector<GlobalCritic>(NUM_THREADS);

	//start both with the same parameters
	auto actor_default_path = CSTRING("networks//" TASK_NAME "//actor_default.nn");
	auto critic_default_path = CSTRING("networks//" TASK_NAME "//critic_default.nn");

	using actor_default_t = decltype(actor_default_path);
	using critic_default_t = decltype(critic_default_path);

	//best path
#ifdef USE_DEMONSTRATIONS
#ifdef USE_RANDOM_SAMPLING
	auto actor_best_path = CSTRING("networks//" TASK_NAME "//random//actor_dem_best.nn");
	auto critic_best_path = CSTRING("networks//" TASK_NAME "//random//critic_dem_best.nn");
#else
	auto actor_best_path = CSTRING("networks//" TASK_NAME "//greedy//actor_dem_best.nn");
	auto critic_best_path = CSTRING("networks//" TASK_NAME "//greedy//critic_dem_best.nn");
#endif
#else
	auto actor_best_path = CSTRING("networks//" TASK_NAME "//actor_nodem_best.nn");
	auto critic_best_path = CSTRING("networks//" TASK_NAME "//critic_nodem_best.nn");
#endif

	using actor_best_t = decltype(actor_best_path);
	using critic_best_t = decltype(critic_best_path);

	float actor_best_score = -INFINITY;

	GlobalActor::save_data<actor_default_t>();
	GlobalCritic::save_data<critic_default_t>();

start:
	{
		srand(0);

		//this is for saving the average score to output every EPISODES_BETWEEN_TESTS training episodes
		Analyzer::sample_size = 50;

		float start_t = 0;
		float last_epoch_t = 0;

		actor_threads = vector<GlobalActor>(NUM_THREADS);
		critic_threads = vector<GlobalCritic>(NUM_THREADS);
		randoms = vector<random_device>(NUM_THREADS);

		//thread-safe random generators
		for (size_t i = 0; i < NUM_THREADS; ++i)
		{
			gens.push_back(mt19937{ (size_t)rand() });
		}

		//start environment
		SimEnvi sim{};

		//prevent loading on each call
		mjState* demonstration_states[NUM_DEMONSTRATIONS];
		mjOneBody* demonstration_onebodies[NUM_DEMONSTRATIONS][nbody];

#ifdef USE_DEMONSTRATIONS
		for (size_t i = 0; i < NUM_DEMONSTRATIONS; ++i)
		{
			demonstration_states[i] = new mjState();
			for (size_t j = 0; j < nbody; ++j)
			{
				demonstration_onebodies[i][j] = new mjOneBody();
				demonstration_onebodies[i][j]->bodyid = j;
			}
		}

		for (size_t i = 0; i < NUM_DEMONSTRATIONS; ++i)
			sim.load_state("demonstrations//" + std::string(TASK_NAME) + "//save" + to_string(i) + ".dat", demonstration_states[i], demonstration_onebodies[i]);

#ifdef USE_RANDOM_SAMPLING
		auto actor_save_path = CSTRING("networks//" TASK_NAME "//random//actor_dem.nn");
		auto critic_save_path = CSTRING("networks//" TASK_NAME "//random//critic_dem.nn");
#else
		auto actor_save_path = CSTRING("networks//" TASK_NAME "//greedy//actor_dem.nn");
		auto critic_save_path = CSTRING("networks//" TASK_NAME "//greedy//critic_dem.nn");
#endif
#else
		auto actor_save_path = CSTRING("networks//" TASK_NAME "//actor_nodem.nn");
		auto critic_save_path = CSTRING("networks//" TASK_NAME "//critic_nodem.nn");
#endif

		using actor_save_t = decltype(actor_save_path);
		using critic_save_t = decltype(critic_save_path);

#ifdef MUJOCO_PRO
		std::array<SimEnvi, NUM_THREADS> sims{};
#endif

		//print info about environment and setup

		cout << "SIMULATION SETTINGS:" << endl;
		cout << "\tTask: " << TASK_NAME << endl;
		cout << "\tAlgorithm: ";
#ifdef USE_DEMONSTRATIONS
#ifdef USE_RANDOM_SAMPLING
		cout << "HCR-A3C" << endl;
#else
		cout << "GHCR-A3C" << endl;
#endif
#else
		cout << "Generic A3C" << endl;
#endif
#ifndef USE_PREPROCESSED_INPUTS
		cout << "\tUsing raw inputs" << endl;
#endif
#ifdef MUJOCO_PRO
		cout << "Running on MuJoCo Pro" << endl;
#else
		cout << "Running on MuJoCo Haptix" << endl;
#endif

		sim.print_body_ids();
		sim.print_joint_ids();
		sim.print_mjinfo();

		if (!TRAINING)
		{
			GlobalActor::load_data<actor_best_t>();
			GlobalCritic::load_data<critic_best_t>();
		}

		size_t T = 0;
		size_t ep = 0;
		int num_epoch = -1;
		start_t = clock();
		while (ep < NUM_TRAINING_EPISODES) // stop after NUM_TRAINING_EPISODES episodes
		{
			//divy up samples
			get_samples();

#ifdef USE_OPENMP
#pragma omp parallel num_threads(NUM_THREADS) reduction(+:T)
#else
			for (int i = 0; i < NUM_THREADS; ++i)
#endif

			{
				for (int j = 0; j < EPISODES_BETWEEN_TESTS && TRAINING; ++j)
				{
#ifdef USE_OPENMP
					int i = omp_get_thread_num();
#endif

#ifndef MUJOCO_PRO
					SimEnvi& environment = sim;
#else
					SimEnvi& environment = sims[i];
#endif

					environment.reset();

#ifdef USE_DEMONSTRATIONS
#ifdef USE_RANDOM_SAMPLING
					int num = randoms[i]() % NUM_DEMONSTRATIONS;
#else
					int num = NUM_DEMONSTRATIONS - 1 - (ep % NUM_DEMONSTRATIONS);//NUM_DEMONSTRATIONS - 1 - (ep / (NUM_TRAINING_EPISODES / NUM_DEMONSTRATIONS)); //don't simplify so integer arithmetic doesn't optimize away
#endif
					environment.set_current_state(demonstration_states[num], demonstration_onebodies[num]);
#endif

					//run a3c
					float score = A3C(T, i, environment);

					//#pragma omp critical
					//cout << score << endl;
#ifdef USE_OPENMP
#pragma omp single
#endif
					{
						++ep;

						#ifdef FIXED_VARIANCE
									if (variance > .005f)
										variance *= .999f;
						#endif

						if (ep % 5 == 0)//anneal weights
						{
							GlobalActor::learning_rate *= .99f;
							GlobalCritic::learning_rate *= .99f;
						}

						//update weights (don't need as a3c is updated?
						//GlobalActor::apply_gradient();
						//GlobalCritic::apply_gradient();
					}
				}

#ifdef USE_OPENMP
#pragma omp barrier
#endif
			}

			if (ep % EPISODES_BETWEEN_TESTS == 0)
			{
				//update from global
#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
#endif
				for (int i = 0; i < NUM_THREADS; ++i)
					GlobalActor::loop_all_layers<actor_thread_reinit, GlobalActor&>(actor_threads[i], 0);

				float avg = 0.0f;
				cout << (clock() - start_t) / CLOCKS_PER_SEC << " sec" << endl;
				cout << "Episode: " << ep << "; TESTING...";
				cout.flush();

				InputFM input{ 0 };

#ifdef USE_OPENMP
#pragma omp parallel for num_threads(NUM_THREADS)
#endif
				for (int tests = 0; tests < Analyzer::sample_size; ++tests)
				{
					int tid = omp_get_thread_num();

#ifndef MUJOCO_PRO
					SimEnvi& environment = sim;
#else
					SimEnvi& environment = sims[tid];
#endif

					environment.reset();

					//run a test
					bool terminal = false;
					float score = 0;
					ActMat act{};
					size_t frame = 0;
					InputFM input{ 0 };

					float reward = 0.0f;
					for (size_t t = 0; !terminal; ++t)
					{
						if (environment.episode_ended())
							terminal = true;

						//discounted reward
						score += pow(DISCOUNT_FACTOR, frame) * reward;

						environment.get_current_state(input);

						//get next action
						actor_threads[tid].discriminate_thread(input);

						//get max act
						act = get_action(actor_threads[tid].get_thread_batch_activations<GlobalActor::last_layer_index>()[0], tid);

						//pass to environment and get reward
						environment.act(act);

						//calculate reward (get new d->xpos etc.)
						reward = environment.get_current_reward();
						++frame;
					}

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
				cout << MEAN_TRANSFORM(actor_threads[0].get_thread_batch_activations<GlobalActor::last_layer_index>()[0][0].at(0, 0)) << ',' << VAR_TRANSFORM(actor_threads[0].get_thread_batch_activations<GlobalActor::last_layer_index>()[0][0].at(1, 0)) << endl;

				//save score and variance
#ifdef SAVE_DATA
#ifdef USE_DEMONSTRATIONS
#ifdef USE_RANDOM_SAMPLING
				Analyzer::save_mean_error("networks//" + std::string(TASK_NAME) + "//random//scores_training_dem.dat");
#else
				Analyzer::save_mean_error("networks//" + std::string(TASK_NAME) + "//greedy//scores_training_dem.dat");
#endif
#else
				Analyzer::save_mean_error("networks//" + std::string(TASK_NAME) + "//scores_training_nodem.dat");
#endif

				//GlobalActor::save_data<actor_save_t>();
				//GlobalCritic::save_data<critic_save_t>();

				if (avg > actor_best_score)
				{
					actor_best_score = avg;
					GlobalActor::save_data<actor_best_t>();
					GlobalCritic::save_data<critic_best_t>();
				}
#endif
			}
		}

		//cleanup data
#ifdef USE_DEMONSTRATIONS
		for (size_t i = 0; i < NUM_DEMONSTRATIONS; ++i)
		{
			delete demonstration_states[i];
			for (size_t j = 0; j < nbody; ++j)
				delete demonstration_onebodies[i][j];
			delete demonstration_onebodies[i];
		}
#endif
	}

	//go to demonstrations

	//reload default params
	GlobalActor::load_data<actor_default_t>();
	GlobalCritic::load_data<critic_default_t>();

	GlobalActor::learning_rate = .0001f;
	GlobalCritic::learning_rate = .0001f;

	//get rid of previous analyzer params
	Analyzer::errors.clear();

	return 0; //todo demonstrations for full cartpole, doesn't replace macro as of now

#ifndef USE_DEMONSTRATIONS
#define USE_DEMONSTRATIONS
#else
#undef USE_DEMONSTRATIONS
#endif

	goto start;

	return 0;
}
