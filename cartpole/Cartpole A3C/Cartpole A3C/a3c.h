#pragma once
#include <algorithm>
#include <omp.h>
#include <random>

#include <imatrix.h>
#include <ilayer.h>
#include <neuralnet.h>

#include "mjcCartpole.h"

////defining the networks

#define MEAN_TRANSFORM(x) x //tanh(x)
#define MEAN_DERIVATIVE(x) 1.0f// - tanh(x) * tanh(x)

#define VAR_TRANSFORM(x) (x > 6 ?  log(1 + exp(6.0f)) : log(1 + exp(x)))
#define VAR_DERIVATIVE(x) (x > 6 ? 1.0f : 1.0f / (1 + exp(-x)))

#define BALANCE 1

//#define USING_LSTM 1 doesn't work rn


//ACTUAL CODE USED HERE
#define NUM_THREADS 4 //works w/ 1, 2, 4 (better w/ more)
#ifdef BALANCE
#define tMAX 500
#else
#define tMAX 300
#endif
#define DISCOUNT_FACTOR .99f

typedef NeuralNet<
	InputLayer<1, 1, 04, 1>,
	//BatchNormalizationLayer<1, 1, 04, 1, MTNN_FUNC_LINEAR>,
	PerceptronFullConnectivityLayer<2, 1, 04, 1, 1, 128, 1, MTNN_FUNC_RELU, true>,
	PerceptronFullConnectivityLayer<3, 1, 128, 1, 1, 200, 1, MTNN_FUNC_RELU, true>,
	PerceptronFullConnectivityLayer<4, 1, 200, 1, 1, 128, 1, MTNN_FUNC_RELU, true>,
#ifdef USING_LSTM
	LSTMLayer<4, 1, 128, 1, 1, 2, 1, tMAX>,
#else
	PerceptronFullConnectivityLayer<4, 1, 128, 1, 1, 2, 1, MTNN_FUNC_LINEAR, true>,
#endif
	PerceptronFullConnectivityLayer<5, 1, 2, 1, 1, 2, 1, MTNN_FUNC_LINEAR, true>, //to transform output
	OutputLayer<5, 1, 2, 1>> GlobalActor;

typedef NeuralNet<
	InputLayer<10, 1, 04, 1>,
	//BatchNormalizationLayer<10, 1, 04, 1, MTNN_FUNC_LINEAR>,
	PerceptronFullConnectivityLayer<20, 1, 04, 1, 1, 128, 1, MTNN_FUNC_RELU, true>,
	PerceptronFullConnectivityLayer<30, 1, 128, 1, 1, 200, 1, MTNN_FUNC_RELU, true>,
	PerceptronFullConnectivityLayer<40, 1, 200, 1, 1, 128, 1, MTNN_FUNC_RELU, true>,
#ifdef USING_LSTM
	LSTMLayer<40, 1, 128, 1, 1, 1, 1, tMAX>,
#else
	PerceptronFullConnectivityLayer<40, 1, 128, 1, 1, 1, 1, MTNN_FUNC_LINEAR, true>,
#endif
	PerceptronFullConnectivityLayer<50, 1, 1, 1, 1, 1, 1, MTNN_FUNC_LINEAR, true>, //to transform output
	OutputLayer<50, 1, 1, 1>> GlobalCritic;

size_t GlobalActor::optimization_method = MTNN_OPT_ADAM;
size_t GlobalActor::loss_function = MTNN_LOSS_CUSTOMTARGETS;
float GlobalActor::learning_rate = .001f;
bool GlobalActor::use_l2_weight_decay = false; //works currently w/o regularization, but large variance/mean (ISN't FIXED BY REGULARIZATION!!)
float GlobalActor::weight_decay_factor = .001f;


size_t GlobalCritic::optimization_method = MTNN_OPT_ADAM;
size_t GlobalCritic::loss_function = MTNN_LOSS_CUSTOMTARGETS;
float GlobalCritic::learning_rate = .001f;
bool GlobalCritic::use_l2_weight_decay = false;
float GlobalCritic::weight_decay_factor = .001f;

typedef Matrix2D<float, 04, 1> InputMat;
typedef FeatureMap<1, 04, 1> InputFM;

std::vector<GlobalActor> actor_threads;
std::vector<GlobalCritic> critic_threads;

std::random_device rd{};
std::mt19937 gen{ 0 };

//#define FIXED_VARIANCE 1
float variance = 1.0f;

//////HogWild!

//updating threads from global and reset gradients
template<size_t l, typename global_net> struct update_params_impl
{
	update_params_impl(global_net& sub_net)
	{
		using layer = global_net::template get_layer<l>;
		//begin weights values
		{
			using t = decltype(layer::weights);
			for (size_t d = 0; d < t::size(); ++d)
			{
				for (size_t i = 0; i < t::rows(); ++i)
				{
					for (size_t j = 0; j < t::cols(); ++j)
					{
						sub_net.get_aux_weights<l>()[d].at(i, j) = layer::weights[d].at(i, j);
						sub_net.get_aux_weights_gradient<l>()[d].at(i, j) = 0;
					}
				}
			}
		}

		//begin biases values
		{
			using t = decltype(layer::biases);
			for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
			{
				for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
				{
					for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
					{
						sub_net.get_aux_biases<l>()[f_0].at(i_0, j_0) = layer::biases[f_0].at(i_0, j_0);
						sub_net.get_aux_biases_gradient<l>()[f_0].at(i_0, j_0) = 0;
					}
				}
			}
		}
	}
};

template<size_t l> using critic_thread_reinit = update_params_impl<l, GlobalCritic>;
template<size_t l> using actor_thread_reinit = update_params_impl<l, GlobalActor>;

//used for global gradient values
std::vector<std::vector<float*>> critic_updates(NUM_THREADS);
std::vector<std::vector<float*>> actor_updates(NUM_THREADS);

//used for gradient values
std::vector<std::vector<float*>> critic_thread_gradients(NUM_THREADS);
std::vector<std::vector<float*>> actor_thread_gradients(NUM_THREADS);

//actually divy up into samples, need to do on master thread
template<size_t l> struct sample_critic_global
{
	//define here so all subclasses have access
	sample_critic_global()
	{
		using layer = GlobalCritic::template get_layer<l>;
		//begin weights values
		{
			using t = decltype(layer::weights);
			for (size_t d = 0; d < t::size(); ++d)
			{
				for (size_t i = 0; i < t::rows(); ++i)
				{
					for (size_t j = 0; j < t::cols(); ++j)
					{
						size_t tid = rand() % NUM_THREADS;
						critic_updates[tid].push_back(&(layer::weights_gradient[d].at(i, j)));
						critic_thread_gradients[tid].push_back(&(critic_threads[tid].get_aux_weights_gradient<l>()[d].at(i, j)));
					}
				}
			}
		}

		//begin biases values
		{
			using t = decltype(layer::biases);
			for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
			{
				for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
				{
					for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
					{
						size_t tid = rand() % NUM_THREADS;
						critic_updates[tid].push_back(&(layer::biases_gradient[f_0].at(i_0, j_0)));
						critic_thread_gradients[tid].push_back(&(critic_threads[tid].get_aux_biases_gradient<l>()[f_0].at(i_0, j_0)));
					}
				}
			}
		}
	}
};
template<size_t l> struct sample_actor_global
{
	//define here so all subclasses have access
	sample_actor_global()
	{
		using layer = GlobalActor::template get_layer<l>;
		//begin weights values
		{
			using t = decltype(layer::weights);
			for (size_t d = 0; d < t::size(); ++d)
			{
				for (size_t i = 0; i < t::rows(); ++i)
				{
					for (size_t j = 0; j < t::cols(); ++j)
					{
						size_t tid = rand() % NUM_THREADS;
						actor_updates[tid].push_back(&(layer::weights_gradient[d].at(i, j)));
						actor_thread_gradients[tid].push_back(&(actor_threads[tid].get_aux_weights_gradient<l>()[d].at(i, j)));
					}
				}
			}
		}

		//begin biases values
		{
			using t = decltype(layer::biases);
			for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
			{
				for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
				{
					for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
					{
						size_t tid = rand() % NUM_THREADS;
						actor_updates[tid].push_back(&(layer::biases_gradient[f_0].at(i_0, j_0)));
						actor_thread_gradients[tid].push_back(&(actor_threads[tid].get_aux_biases_gradient<l>()[f_0].at(i_0, j_0)));
					}
				}
			}
		}
	}
};

void get_samples()
{
	//get rid of old samples
	for (size_t tid = 0; tid < NUM_THREADS; ++tid)
	{
		critic_updates[tid].clear();
		critic_thread_gradients[tid].clear();
		actor_updates[tid].clear();
		actor_thread_gradients[tid].clear();
	}

	//Get critic samples
	GlobalCritic::loop_all_layers<sample_critic_global>(0);
	//get actor samples
	GlobalActor::loop_all_layers<sample_actor_global>(0);
}

////HogWild! updates

//update critic asynchronous according to HogWild!
void update_globals_critic(size_t tid)
{
	auto& update_vec = critic_updates[tid];
	auto& grad_vec = critic_thread_gradients[tid];

	for (size_t i = 0; i < update_vec.size(); ++i)
		*update_vec[i] += *grad_vec[i];// *NUM_THREADS; //stores thread gradient in global gradient
}

//update actor asynchronous according to HogWild!  /////TODO: hogwild |e| updates?
void update_globals_actor(size_t tid)
{
	auto& update_vec = actor_updates[tid];
	auto& grad_vec = actor_thread_gradients[tid];

	for (size_t i = 0; i < update_vec.size(); ++i)
		*update_vec[i] += *grad_vec[i];// *NUM_THREADS; //stores thread gradient in global gradient 
}

float reduce_angle(float angle)
{
	if (angle > 3.1415f)
		angle = reduce_angle((angle -= 2 * 3.1415f));
	else if (angle < -3.1415f)
		angle = reduce_angle((angle += 2 * 3.1415f));
	return angle;
}

float clip(float n, float lower, float upper) 
{
	return max(lower, min(n, upper));
}

float get_action(FeatureMap<1, 2, 1>& output)
{
#ifdef FIXED_VARIANCE
	std::normal_distribution<float> dist{ MEAN_TRANSFORM(output[0].at(0, 0)), sqrt(variance) };
#else
	std::normal_distribution<float> dist{ MEAN_TRANSFORM(output[0].at(0, 0)), sqrt(VAR_TRANSFORM(output[0].at(1, 0)) + .0001f) };
#endif
	float f = dist(gen);
	return f;//clip(f, -1, 1);
}

#define REWARD_FUNC(x) .01f * (-pow(x, 2))

float A3C(size_t& T, int tid, CartpoleSimEnvi& sim)
{
	bool gone_up = false;

#ifdef BALANCE
	gone_up = true;
#endif

	auto& actor = actor_threads[tid];
	auto& critic = critic_threads[tid];

	bool terminal = false;
	float score = 0;
	float act = 0;
	size_t frame = 0;
	InputFM input{ 0 };

	int cart_id = mj_name2id("body", "cart"), cart_joint_id = mj_name2id("joint", "slider"), pole_id = mj_name2id("site", "tip"), pole_joint_id = mj_name2id("joint", "hinge");
	mjState* currentState = new mjState();
	mjOneBody* cart = new mjOneBody();
	mjControl* control = new mjControl();
	cart->bodyid = cart_id;
	mj_get_state(currentState);
	mj_get_onebody(cart);
	mj_get_control(control);

	////training step
	while (!terminal)
	{
		//reset thread gradients and update from global; now applying directly after train won't affect other threads' belief of current weights
		GlobalCritic::loop_all_layers<critic_thread_reinit, GlobalCritic&>(critic, 0);
		GlobalActor::loop_all_layers<actor_thread_reinit, GlobalActor&>(actor, 0);

		float reward = 0.0f;
		float V = 0.0f;
		float theta = 3.141592f - (float)currentState->qpos[pole_joint_id];
		std::vector<std::tuple<InputFM, float, float, float>> mem;
		for (size_t t = 0; !terminal; ++t)
		{
			float t2 = reduce_angle(theta);
			if ((gone_up && abs(t2) > 3.1415f / 2))
				terminal = true;
			if (!gone_up && abs(t2) < 3.1415f / 4)
				gone_up = true;
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
			if (t > 0)
				mem.push_back(std::tuple<InputFM, float, float, float>(input, reward, act, V)); //update if reward is given late
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
			actor.discriminate_thread(input);

			//get critic value
			critic.discriminate_thread(input);
			V = critic.get_thread_batch_activations<GlobalCritic::last_layer_index>()[0][0].at(0, 0);
			if (t == 0 && tid == 0)
				std::cout << "critic:" << V << std::endl;

			//get max act
			act = get_action(actor.get_thread_batch_activations<GlobalActor::last_layer_index>()[0]);

			//pass to environment and get reward
			control->ctrl[0] = act;
			mj_set_control(control);

			for (int i = 0; i < 3; ++i)
				mj_step();

#ifdef BALANCE
			reward = 0;
#else
			//calculate reward (get new d->xpos etc.)
			//t2 = reduce_angle(theta);
			//reward = REWARD_FUNC(t2);

			reward = 0;
#endif			
			++frame;
		}
		float R = 0;
		if (!terminal)
			R = V; //bootstrap

		//update gradients
		for (int t = mem.size() - 1; t > -1; t--)
		{
			auto& state = std::get<0, InputFM, float, float, float>(mem[t]);
			float& r = std::get<1, InputFM, float, float, float>(mem[t]);
			float& act = std::get<2, InputFM, float, float, float>(mem[t]);
			float& v = std::get<3, InputFM, float, float, float>(mem[t]);

			R = DISCOUNT_FACTOR * R + r;//fails when r!= 0?

			//train critic (straight forward)
			//train actor

			//start by loading the states into the actor/critic
			actor.discriminate_thread(state);
			critic.discriminate_thread(state);

			float out = actor.get_thread_batch_activations<GlobalActor::last_layer_index>()[0][0].at(1, 0);
			float out1 = actor.get_thread_batch_activations<GlobalActor::last_layer_index>()[0][0].at(0, 0);
#ifdef FIXED_VARIANCE
			float var = variance;
#else
			float var = VAR_TRANSFORM(out) + .0001f;
#endif
			float mu = MEAN_TRANSFORM(out1);
			float zstd = (act - mu) / var;
			float td = (R - v);
			float prob = exp(-(act - mu) * (act - mu) / var / 2) / sqrt(2 * 3.1415f * var);
			//no act plus descent causes great w/o variance, but badddd variance
#ifdef FIXED_VARIANCE
			FeatureMap<1, 2, 1> lbls({ -(td * zstd) * MEAN_DERIVATIVE(out1), 0 });//neg for descent
#else
			FeatureMap<1, 2, 1> lbls({ -(td * zstd) * MEAN_DERIVATIVE(out1), -(td * (zstd * zstd - 1.0f / var) / 2 - .0001f / 2 / var) * VAR_DERIVATIVE(out) });//neg for descent
#endif

			actor.train_thread(true, state, lbls);
			critic.train_thread(true, state, FeatureMap<1, 1, 1>(-td)); //deriv flips sign

			//pop off for bptt
#ifdef USING_LSTM
			GlobalActor::template get_layer<4>::cell_states.pop_back();
			GlobalCritic::template get_layer<4>::cell_states.pop_back();
			GlobalActor::template get_layer<4>::hidden_states.pop_back();
			GlobalCritic::template get_layer<4>::hidden_states.pop_back();
			GlobalActor::template get_layer<4>::forget_states.pop_back();
			GlobalCritic::template get_layer<4>::forget_states.pop_back();
			GlobalActor::template get_layer<4>::influence_states.pop_back();
			GlobalCritic::template get_layer<4>::influence_states.pop_back();
			GlobalActor::template get_layer<4>::activation_states.pop_back();
			GlobalCritic::template get_layer<4>::activation_states.pop_back();
			GlobalActor::template get_layer<4>::output_states.pop_back();
			GlobalCritic::template get_layer<4>::output_states.pop_back();
#endif
		}

		//reset derivs for bptt
#ifdef USING_LSTM
		GlobalActor::template get_layer<4>::cell_state_deriv = { 0 };
		GlobalCritic::template get_layer<4>::cell_state_deriv = { 0 };
#endif

		//update values
		update_globals_critic(tid);
		update_globals_actor(tid);

#pragma omp single nowait
		{
			GlobalActor::apply_gradient();
			GlobalCritic::apply_gradient();
		}
	}

	delete currentState;
	delete cart;
	delete control;

	T += frame;

#pragma omp barrier
	return score;
}
