#pragma once
#include <algorithm>
#include <stdlib.h>
#include <omp.h>
#include <random>

#include <imatrix.h>
#include <ilayer.h>
#include <neuralnet.h>

#include "mjcEnvironment.h"

//ACTUAL CODE USED HERE
#define NUM_THREADS 8

#ifdef MUJOCO_PRO
#define USE_OPENMP
#endif

#define tMAX MAX_FRAMES
#define DISCOUNT_FACTOR .99f

////defining the networks

#define MEAN_TRANSFORM(x) x //tanh(x)
#define MEAN_DERIVATIVE(x) 1.0f// - tanh(x) * tanh(x)

#define VAR_TRANSFORM(x) (x > 6 ?  log(1 + exp(6.0f)) : (x < -6 ? log(1 + exp(-6.0f)) : log(1 + exp(x))))
#define VAR_DERIVATIVE(x) (x > 6 ? 1.0f : (x < -6 ? 1.0f / (1 + exp(6.0f)) : 1.0f / (1 + exp(-x))))
//#define USING_LSTM 1 //todo doesn't work

typedef NeuralNet<
	InputLayer<1, 1, nq + nv, 1>,
	//BatchNormalizationLayer<1, 1, nq + nv, 1, MTNN_FUNC_LINEAR>,
	PerceptronFullConnectivityLayer<2, 1, nq + nv, 1, 1, 128, 1, MTNN_FUNC_RELU, true>,
	PerceptronFullConnectivityLayer<3, 1, 128, 1, 1, 200, 1, MTNN_FUNC_RELU, true>,
	PerceptronFullConnectivityLayer<4, 1, 200, 1, 1, 128, 1, MTNN_FUNC_RELU, true>,
#ifdef USING_LSTM
	LSTMLayer<4, 1, 128, 1, 1, nu, 2, tMAX>,
#else
	PerceptronFullConnectivityLayer<4, 1, 128, 1, 1, nu, 2, MTNN_FUNC_LINEAR, true>,
#endif
	PerceptronFullConnectivityLayer<5, 1, nu, 2, 1, nu, 2, MTNN_FUNC_LINEAR, true>, //to transform output
	OutputLayer<5, 1, nu, 2>> GlobalActor;

typedef NeuralNet<
	InputLayer<10, 1, nq + nv, 1>,
	//BatchNormalizationLayer<10, 1, nq + nv, 1, MTNN_FUNC_LINEAR>,
	PerceptronFullConnectivityLayer<20, 1, nq + nv, 1, 1, 128, 1, MTNN_FUNC_RELU, true>,
	PerceptronFullConnectivityLayer<30, 1, 128, 1, 1, 200, 1, MTNN_FUNC_RELU, true>,
	PerceptronFullConnectivityLayer<40, 1, 200, 1, 1, 128, 1, MTNN_FUNC_RELU, true>,
#ifdef USING_LSTM
	LSTMLayer<40, 1, 128, 1, 1, 1, 1, tMAX>,
#else
	PerceptronFullConnectivityLayer<40, 1, 128, 1, 1, 1, 1, MTNN_FUNC_LINEAR, true>,
#endif
	PerceptronFullConnectivityLayer<50, 1, 1, 1, 1, 1, 1, MTNN_FUNC_LINEAR, true>, //to transform output
	OutputLayer<50, 1, 1, 1>> GlobalCritic;


FeatureMap<1, 128, 200> PerceptronFullConnectivityLayer<3, 1, 200, 1, 1, 128, 1, MTNN_FUNC_RELU, true>::weights = { -1.0f / 141.f, 1.0f / 141.f };
FeatureMap<1, nu * 2, 128> PerceptronFullConnectivityLayer<4, 1, 128, 1, 1, nu, 2, MTNN_FUNC_LINEAR, true>::weights = { -1.0f / 112.f, 1.0f / 112.f };

FeatureMap<1, 128, 200> PerceptronFullConnectivityLayer<30, 1, 200, 1, 1, 128, 1, MTNN_FUNC_RELU, true>::weights = { -1.0f / 14.1f, 1.0f / 14.1f };
FeatureMap<1, 1, 128> PerceptronFullConnectivityLayer<40, 1, 128, 1, 1, 1, 1, MTNN_FUNC_LINEAR, true>::weights = { -1.0f / 11.2f, 1.0f / 11.2f };

size_t GlobalActor::optimization_method = MTNN_OPT_ADAM;// MTNN_OPT_RMSPROP;
size_t GlobalActor::loss_function = MTNN_LOSS_CUSTOMTARGETS;
float GlobalActor::learning_rate = .001f;
bool GlobalActor::use_l2_weight_decay = false;
float GlobalActor::weight_decay_factor = .001f;


size_t GlobalCritic::optimization_method = MTNN_OPT_ADAM;// MTNN_OPT_RMSPROP;
size_t GlobalCritic::loss_function = MTNN_LOSS_CUSTOMTARGETS;
float GlobalCritic::learning_rate = .001f;
bool GlobalCritic::use_l2_weight_decay = false;
float GlobalCritic::weight_decay_factor = .001f;

typedef Matrix2D<float, nq + nv, 1> InputMat;
typedef FeatureMap<1, nq + nv, 1> InputFM;
typedef Matrix2D<float, nu, 2> OutputMat;
typedef FeatureMap<1, nu, 2> OutputFM;
typedef Matrix2D<float, nu, 1> ActMat;

std::vector<std::random_device> randoms{};
std::vector<std::mt19937> gens{};

std::vector<GlobalActor> actor_threads;
std::vector<GlobalCritic> critic_threads;

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

/*clipping
template<size_t l, typename global_net> struct get_gradient_norm_impl
{
	get_gradient_norm_impl(float* norm)
	{
		using layer = global_net::template get_layer<l>;

		//begin weights values
		{
			using t = decltype(layer::weights);
			for (size_t d = 0; d < t::size(); ++d)
				for (size_t i = 0; i < t::rows(); ++i)
					for (size_t j = 0; j < t::cols(); ++j)
						*norm += pow(layer::weights_gradient[d].at(i, j), 2);
		}

		//begin biases values
		{
			using t = decltype(layer::biases);
			for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
				for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
					for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
						*norm += pow(layer::biases_gradient[f_0].at(i_0, j_0), 2);
		}
	}
};

template<size_t l, typename global_net> struct get_thread_gradient_norm_impl
{
	get_thread_gradient_norm_impl(global_net& sub_net, float* norm)
	{
		using layer = global_net::template get_layer<l>;

		//begin weights values
		{
			using t = decltype(layer::weights);
			for (size_t d = 0; d < t::size(); ++d)
				for (size_t i = 0; i < t::rows(); ++i)
					for (size_t j = 0; j < t::cols(); ++j)
						*norm += pow(sub_net.get_aux_weights_gradient<l>()[d].at(i, j), 2);
		}

		//begin biases values
		{
			using t = decltype(layer::biases);
			for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
				for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
					for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
						*norm += pow(sub_net.get_aux_biases_gradient<l>()[f_0].at(i_0, j_0), 2);
		}
	}
};

template<size_t l> using actor_gradient_norm = get_gradient_norm_impl<l, GlobalActor>;
template<size_t l> using critic_gradient_norm = get_gradient_norm_impl<l, GlobalCritic>;

template<size_t l> using actor_thread_gradient_norm = get_thread_gradient_norm_impl<l, GlobalActor>;
template<size_t l> using critic_thread_gradient_norm = get_thread_gradient_norm_impl<l, GlobalCritic>;

template<size_t l, typename global_net> struct clip_gradients_impl
{
	clip_gradients_impl(float norm, float max)
	{
		float clip_coeff = max / (norm + 1e-6f);

		if (clip_coeff < 1.0f)
		{
			using layer = global_net::template get_layer<l>;

			//begin weights values
			{
				using t = decltype(layer::weights);
				for (size_t d = 0; d < t::size(); ++d)
					for (size_t i = 0; i < t::rows(); ++i)
						for (size_t j = 0; j < t::cols(); ++j)
							layer::weights_gradient[d].at(i, j) *= clip_coeff;
			}

			//begin biases values
			{
				using t = decltype(layer::biases);
				for (size_t f_0 = 0; f_0 < t::size(); ++f_0)
					for (size_t i_0 = 0; i_0 < t::rows(); ++i_0)
						for (size_t j_0 = 0; j_0 < t::cols(); ++j_0)
							layer::biases_gradient[f_0].at(i_0, j_0) *= clip_coeff;
			}
		}
	}
};

template<size_t l> using actor_clip_gradients = clip_gradients_impl<l, GlobalActor>;
template<size_t l> using critic_clip_gradients = clip_gradients_impl<l, GlobalCritic>;*/

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

//update actor asynchronous according to HogWild!
void update_globals_actor(size_t tid)
{
	auto& update_vec = actor_updates[tid];
	auto& grad_vec = actor_thread_gradients[tid];

	for (size_t i = 0; i < update_vec.size(); ++i)
		*update_vec[i] += *grad_vec[i];// *NUM_THREADS; //stores thread gradient in global gradient 
}

ActMat get_action(OutputFM& output, int tid)
{
	ActMat out{};
	for (size_t i = 0; i < nu; ++i)
	{
#ifdef FIXED_VARIANCE
		std::normal_distribution<float> dist{ MEAN_TRANSFORM(output[0].at(0, 0)), sqrt(variance) };
#else
		std::normal_distribution<float> dist{ MEAN_TRANSFORM(output[0].at(0, 0)), sqrt(VAR_TRANSFORM(output[0].at(1, 0)) + .0001f) };
#endif
		float f = dist(gens[tid]);
		out.at(i, 0) = f;
	}
	return out;
}

inline float A3C(size_t& T, int tid, SimEnvi& sim)
{
	auto& actor = actor_threads[tid];
	auto& critic = critic_threads[tid];

	//fix sizes
	while (actor.get_thread_batch_activations<0>().size() != tMAX) //fix sizes
	{
		if (actor.get_thread_batch_activations<0>().size() > tMAX)
			GlobalActor::loop_all_layers<GlobalActor::remove_thread_batch_activations, GlobalActor&>(actor, 0);
		else
			GlobalActor::loop_all_layers<GlobalActor::add_thread_batch_activations, GlobalActor&>(actor, 0);
	}
	while (critic.get_thread_batch_activations<0>().size() != tMAX) //fix sizes
	{
		if (critic.get_thread_batch_activations<0>().size() > tMAX)
			GlobalCritic::loop_all_layers<GlobalCritic::remove_thread_batch_activations, GlobalCritic&>(critic, 0);
		else
			GlobalCritic::loop_all_layers<GlobalCritic::add_thread_batch_activations, GlobalCritic&>(critic, 0);
	}

	while (actor.get_thread_batch_out_derivs<0>().size() != tMAX) //fix sizes
	{
		if (actor.get_thread_batch_out_derivs<0>().size() > tMAX)
			GlobalActor::loop_all_layers<GlobalActor::remove_thread_batch_out_derivs, GlobalActor&>(actor, 0);
		else
			GlobalActor::loop_all_layers<GlobalActor::add_thread_batch_out_derivs, GlobalActor&>(actor, 0);
	}
	while (critic.get_thread_batch_out_derivs<0>().size() != tMAX) //fix sizes
	{
		if (critic.get_thread_batch_out_derivs<0>().size() > tMAX)
			GlobalCritic::loop_all_layers<GlobalCritic::remove_thread_batch_out_derivs, GlobalCritic&>(critic, 0);
		else
			GlobalCritic::loop_all_layers<GlobalCritic::add_thread_batch_out_derivs, GlobalCritic&>(critic, 0);
	}

	GlobalActor::loop_all_layers<GlobalActor::reset_thread_feature_maps, GlobalActor&>(actor, 0);
	GlobalCritic::loop_all_layers<GlobalCritic::reset_thread_feature_maps, GlobalCritic&>(critic, 0);

	bool terminal = false;
	float score = 0;
	ActMat act{};
	size_t frame = 0;
	InputFM input{ 0 };

	////training step
	while (!terminal)
	{
		//reset thread gradients and update from global
		GlobalActor::loop_all_layers<actor_thread_reinit, GlobalActor&>(actor, 0);
		GlobalCritic::loop_all_layers<critic_thread_reinit, GlobalCritic&>(critic, 0);

		float reward = 0.0f;
		float V = 0.0f;
		std::vector<std::tuple<InputFM, float, ActMat, float>> mem;

		for (size_t t = 0; !terminal; ++t)
		{
			if (sim.episode_ended() || frame >= tMAX - 1)
				terminal = true;

			score += reward;
			if (t > 0)
				mem.push_back(std::tuple<InputFM, float, ActMat, float>(input, reward, act, V)); //update if reward is given late
			if (t >= tMAX || terminal)
				break;

			sim.get_current_state(input);

			//get next action
			actor.discriminate_thread(input, t);

			//get value
			critic.discriminate_thread(input, t);
			V = critic.get_thread_batch_activations<GlobalCritic::last_layer_index>()[t][0].at(0, 0);

			//get act
			act = get_action(actor.get_thread_batch_activations<GlobalActor::last_layer_index>()[t], tid);

			//pass to environment and get reward
			sim.act(act);

			//calculate reward (get new d->xpos etc.)
			reward = sim.get_current_reward();

			/*std::cout << "state:";
			for (int i = 0; i < nq + nv; ++i)
				std::cout << input[0].at(i, 0) << ", ";
			std::cout << "reward:" << reward << std::endl;

			for (int i = 0; i < nu; ++i)
				std::cout << MEAN_TRANSFORM(actor.get_thread_batch_activations<GlobalActor::last_layer_index>()[0][0].at(i, 0)) << ',' << VAR_TRANSFORM(actor.get_thread_batch_activations<GlobalActor::last_layer_index>()[0][0].at(i, 1)) << ";";
			std::cout << std::endl;
			for (int i = 0; i < nu; ++i)
				std::cout << act.at(i, 0) << ';';
			std::cout << std::endl;*/

			++frame;
		}
		float R = 0;
		float gae = 0;
		if (!terminal)
			R = V; //bootstrap

		//update gradients
		std::vector<OutputFM> lbls = std::vector<OutputFM>(tMAX);
		std::vector<FeatureMap<1, 1, 1>> criticLbls = std::vector<FeatureMap<1, 1, 1>>(tMAX);
		std::vector<InputFM> states = std::vector<InputFM>(tMAX);
		for (int t = mem.size() - 1; t > -1; t--)
		{
			auto& state = std::get<0, InputFM, float, ActMat, float>(mem[t]);
			float& r = std::get<1, InputFM, float, ActMat, float>(mem[t]);
			auto& act = std::get<2, InputFM, float, ActMat, float>(mem[t]);
			auto& v = std::get<3, InputFM, float, ActMat, float>(mem[t]);

			//future discounted rewards
			R = DISCOUNT_FACTOR * R + r;

			//Generalized Advantage Estimation
			float delta_t = r + DISCOUNT_FACTOR * V - v;
			gae = gae * DISCOUNT_FACTOR + gae;
			V = v; //update for next step

#ifndef USE_OPENMP
			//if (t == 0 && tid == 0 && frame <= tMAX)
			//	std::cout << "Start state: V=" << v << ", R=" << R << std::endl;
#endif

			//train actor+critic
			float td = (R - v); //should incorporate future?
			//if (tid == 0 && t == 0)
			//	std::cout << td << std::endl;
			float criticLbl = -td;
			for (size_t i = 0; i < nu; ++i)
			{
				//tenatively correct?
				float out_act = actor.get_thread_batch_activations<GlobalActor::last_layer_index>()[t][0].at(i, 0);
				float out_var = actor.get_thread_batch_activations<GlobalActor::last_layer_index>()[t][0].at(i, 1);
				float mu = MEAN_TRANSFORM(out_act);
				float var = VAR_TRANSFORM(out_var);

				float zstd = (act.at(i, 0) - mu) / var;


				//derivatives
				//lbls[t][0].at(i, 0) = -(gae * zstd) * MEAN_DERIVATIVE(out_act);
				lbls[t][0].at(i, 0) = -(td * zstd) * MEAN_DERIVATIVE(out_act);

#ifndef FIXED_VARIANCE
				//lbls[t][0].at(i, 1) = -(gae * (zstd * zstd - 1.0f / var) / 2 - .0001f / 2 / var) * VAR_DERIVATIVE(out_var);
				lbls[t][0].at(i, 1) = -(td * (zstd * zstd - 1.0f / var) / 2 - .0001f / 2 / var) * VAR_DERIVATIVE(out_var);
#endif

				//std::cout << lbls[t][0].at(i, 0) << ',' << lbls[t][0].at(i, 1) << std::endl;
			}
			criticLbls[t][0].at(0, 0) = criticLbl;
			states[t] = state;

			//pop off for bptt
//#ifdef USING_LSTM
//			GlobalActor::template get_layer<4>::cell_states.pop_back();
//			GlobalCritic::template get_layer<4>::cell_states.pop_back();
//			GlobalActor::template get_layer<4>::hidden_states.pop_back();
//			GlobalCritic::template get_layer<4>::hidden_states.pop_back();
//			GlobalActor::template get_layer<4>::forget_states.pop_back();
//			GlobalCritic::template get_layer<4>::forget_states.pop_back();
//			GlobalActor::template get_layer<4>::influence_states.pop_back();
//			GlobalCritic::template get_layer<4>::influence_states.pop_back();
//			GlobalActor::template get_layer<4>::activation_states.pop_back();
//			GlobalCritic::template get_layer<4>::activation_states.pop_back();
//			GlobalActor::template get_layer<4>::output_states.pop_back();
//			GlobalCritic::template get_layer<4>::output_states.pop_back();
//#endif
		}

		//accumulate gradients
		actor.train_batch_thread(states, lbls);
		critic.train_batch_thread(states, criticLbls);

		////clip gradients to prevent exploding (TODO: investigate why labels are exploding?)
		//float norm = 0.0f;
		//for (size_t i = 0; i < actor_thread_gradients[tid].size(); ++i)
		//	norm += pow(*actor_thread_gradients[tid][i], 2);
		//for (size_t i = 0; i < critic_thread_gradients[tid].size(); ++i)
		//	norm += pow(*critic_thread_gradients[tid][i], 2);

		////std::cout << norm << std::endl;

		//float clip_coeff = 40.0f / norm;
		//if (clip_coeff < 1)
		//{
		//	for (size_t i = 0; i < actor_thread_gradients[tid].size(); ++i)
		//		*actor_thread_gradients[tid][i] *= clip_coeff;
		//	for (size_t i = 0; i < critic_thread_gradients[tid].size(); ++i)
		//		*critic_thread_gradients[tid][i] *= clip_coeff;
		//}

		//reset derivs
#ifdef USING_LSTM
		GlobalActor::template get_layer<4>::cell_state_deriv = { 0 };
		GlobalCritic::template get_layer<4>::cell_state_deriv = { 0 };
#endif

		//update global gradient values
		update_globals_critic(tid);
		update_globals_actor(tid);		

		//apply gradients
//#pragma omp single nowait
		{
			GlobalActor::apply_gradient();
			GlobalCritic::apply_gradient();
		}
	}

	T += frame;

	return score;
}