#include <conio.h>
#include <time.h>

#include "a3c.h"
#include "mujocoBaxter.h"

#include <neuralnetanalyzer.h>

using namespace std;

//#define USE_DEMONSTRATIONS 1

typedef NeuralNetAnalyzer<GlobalActor> Analyzer;

#define NUM_DEMONSTRATIONS 78

#define TRAINING true

int main(int argc, char* argv[])
{
	//srand(clock());
	//SimEnvi sim{};
	//sim.print_body_ids();
	//sim.print_joint_ids();

	////creating/testing demonstrations
	/*mjState* save_state = new mjState();
	mjOneBody* save_bodies[nbody];
	for (int i = 0; i < nbody; ++i)
	{
		save_bodies[i] = new mjOneBody();
		save_bodies[i]->bodyid = i;
	}
	int i = 0;
	
	SimEnvi sim{};
	while (true)
	{
		sim.get_current_state(save_state, save_bodies);
		if (sim.get_current_reward() > 0.0f)
			cout << "COMPLETED!" << endl;
		sim.save_state("demonstrations//save" + to_string(i) + ".dat", save_state, save_bodies);
		cout << "Frame saved. Press enter to save a frame again." << endl;
		_getche();
		++i;
	}*/

	////loading
	//start environment
	/*SimEnvi sim{};
	int i = 0;
	while (true)
	{
		sim.load_state("demonstrations//save" + to_string(i) + ".dat", sim.m_state, sim.m_bodies);
		sim.set_current_state(sim.m_state, sim.m_bodies);
		if (sim.get_current_reward() > 0.0f)
			cout << "COMPLETED!" << endl;
		cout << i << "th frame loaded. Press enter to load the next frame." << endl;
		_getche();
		++i;
	}*/

	float start_t = 0;
	float last_epoch_t = 0;

	actor_threads = vector<GlobalActor>(NUM_THREADS);
	critic_threads = vector<GlobalCritic>(NUM_THREADS);

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
		SimEnvi sim{};

		if (!TRAINING)
		{
			GlobalActor::load_data<actor_save_t>();
			GlobalCritic::load_data<critic_save_t>();
		}

		size_t T = 0;
		size_t ep = 1;
		int num_epoch = -1;
		start_t = clock();
		while (true && ep < 3000) // stop after 2000 episodes
		{
			//divy up samples
			get_samples();

#pragma omp parallel num_threads(NUM_THREADS) reduction(+:T) shared(actor_threads, critic_threads)
			for (int i = 0; i < NUM_THREADS && TRAINING; ++i)
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
			cout << "Actuators:";
			for (int i = 0; i < nu; ++i)
				cout << MEAN_TRANSFORM(actor_threads[0].get_thread_batch_activations<GlobalActor::last_layer_index>()[0][0].at(i, 0)) << ',' << VAR_TRANSFORM(actor_threads[0].get_thread_batch_activations<GlobalActor::last_layer_index>()[0][0].at(i, 1)) << ";";
			cout << endl;

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

			if (ep % 100 == 0)
			{
				float avg = 0.0f;
				cout << (clock() - start_t) / CLOCKS_PER_SEC << " sec" << endl;
				cout << "TESTING...";
				cout.flush();

				InputFM input{ 0 };

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
					ActMat act{};
					size_t frame = 0;
					InputFM input{ 0 };

					float reward = 0.0f;
					for (size_t t = 0; !terminal; ++t)
					{
						if (reward > 0 || sim.m_state->time > 50)
							terminal = true;
						/*if (terminal)
						reward = -.5f;*/

						score += reward;

						sim.get_current_state(input);

						//get next action
						GlobalActor::discriminate(input);

						//get max act
						act = get_action(GlobalActor::template get_batch_activations<GlobalActor::last_layer_index>()[0]);
						//for (int i = 0; i < nu; ++i)
						//	act.at(i, 0) = GlobalActor::template get_batch_activations<GlobalActor::last_layer_index>()[0][0].at(i, 0);

						//pass to environment and get reward
						sim.act(act);

						//calculate reward (get new d->xpos etc.)
						reward = sim.get_current_reward();
						++frame;
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
#ifndef USE_DEMONSTRATIONS
#define USE_DEMONSTRATIONS
#else
#undef USE_DEMONSTRATIONS
#endif // !USE_DEMONSTRATIONS

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
		SimEnvi sim{};

		if (!TRAINING)
		{
			GlobalActor::load_data<actor_save_t>();
			GlobalCritic::load_data<critic_save_t>();
		}

		size_t T = 0;
		size_t ep = 1;
		int num_epoch = -1;
		start_t = clock();
		while (true && ep < 3000) // stop after 2000 episodes
		{
			//divy up samples
			get_samples();

#pragma omp parallel num_threads(NUM_THREADS) reduction(+:T) shared(actor_threads, critic_threads)
			for (int i = 0; i < NUM_THREADS && TRAINING; ++i)
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

			if (ep % 100 == 0)
			{
				float avg = 0.0f;
				cout << (clock() - start_t) / CLOCKS_PER_SEC << " sec" << endl;
				cout << "TESTING...";
				cout.flush();

				InputFM input{ 0 };

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
					ActMat act{};
					size_t frame = 0;
					InputFM input{ 0 };

					float reward = 0.0f;
					for (size_t t = 0; !terminal; ++t)
					{
						if (reward > 0 || sim.m_state->time > 50)
							terminal = true;
						/*if (terminal)
						reward = -.5f;*/

						score += reward;

						sim.get_current_state(input);

						//get next action
						GlobalActor::discriminate(input);

						//get max act
						act = get_action(GlobalActor::template get_batch_activations<GlobalActor::last_layer_index>()[0]);
						//for (int i = 0; i < nu; ++i)
						//	act.at(i, 0) = GlobalActor::template get_batch_activations<GlobalActor::last_layer_index>()[0][0].at(i, 0);

						//pass to environment and get reward
						sim.act(act);

						//calculate reward (get new d->xpos etc.)
						reward = sim.get_current_reward();
						++frame;
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

				GlobalActor::save_data<actor_save_t>();
				GlobalCritic::save_data<critic_save_t>();
			}

			++ep;
		}
	}

	return 0;
}