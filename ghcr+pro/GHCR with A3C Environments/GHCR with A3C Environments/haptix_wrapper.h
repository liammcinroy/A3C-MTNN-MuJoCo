#pragma once

/*
This wrapper allows use of haptix at the same time as pro
*/

#define mjMAXSZ 1000

	/****************************************************************************************

	PART 2:  Native API  (prefix 'mj')

	This API provides more complete access to the simulator and does not make
	assumptions about the model structure, except for the maximum array sizes.
	It aims to maximize the benefits of using the simulator.

	The simple and native APIs can be mixed.

	Unlike the simple API where the user is expected to know the sizes of the
	variable-size arrays (by calling hx_robot_info), here the sizes of the
	variable-size arrays are included in the structures containing the arrays.

	****************************************************************************************/

	// state of the dynamical system in generalized coordinates
	struct _mjState
	{
		int nq;                             // number of generalized positions
		int nv;                             // number of generalized velocities
		int na;                             // number of actuator activations
		float time;                         // simulation time
		float qpos[mjMAXSZ];                // generalized positions
		float qvel[mjMAXSZ];                // generalized velocities
		float act[mjMAXSZ];                 // actuator activations
	};
	typedef struct _mjState mjState;


	// control signals
	struct _mjControl
	{
		int nu;                             // number of actuators
		float time;                         // simulation time
		float ctrl[mjMAXSZ];                // control signals
	};
	typedef struct _mjControl mjControl;


	// applied forces
	struct _mjApplied
	{
		int nv;                             // number of generalized velocities
		int nbody;                          // number of bodies
		float time;                         // simulation time
		float qfrc_applied[mjMAXSZ];        // generalized forces
		float xfrc_applied[mjMAXSZ][6];     // Cartesian forces and torques applied to bodies
	};
	typedef struct _mjApplied mjApplied;


	// detailed information about one body
	struct _mjOneBody
	{
		int bodyid;                         // body id, provided by user

											// get only
		int isfloating;                     // 1 if body is floating, 0 otherwise
		float time;                         // simulation time
		float linacc[3];                    // linear acceleration
		float angacc[3];                    // angular acceleration
		float contactforce[3];              // net force from all contacts on this body

											// get for all bodies; set for floating bodies only
											//  (setting the state of non-floating bodies would require inverse kinematics)
		float pos[3];                       // position
		float quat[4];                      // orientation quaternion
		float linvel[3];                    // linear velocity
		float angvel[3];                    // angular velocity

											// get and set for all bodies 
											//  (modular access to the same data as provided by mjApplied.xfrc_applied)
		float force[3];                     // Cartesian force applied to body CoM
		float torque[3];                    // Cartesian torque applied to body
	};
	typedef struct _mjOneBody mjOneBody;


	// Cartesian positions and orientations of mocap bodies (treated as constant by simulator)
	struct _mjMocap
	{
		int nmocap;                         // number of mocap bodies
		float time;                         // simulation time
		float pos[mjMAXSZ][3];              // positions
		float quat[mjMAXSZ][4];             // quaternion orientations
	};
	typedef struct _mjMocap mjMocap;


	//------------------------- Quantities that can be GET only -----------------------------

	// main output of forward dynamics; used internally to integrate the state
	struct _mjDynamics
	{
		int nv;                             // number of generalized velocities
		int na;                             // number of actuator activations
		float time;                         // simulation time
		float qacc[mjMAXSZ];                // generalized accelerations
		float actdot[mjMAXSZ];              // time-derivatives of actuator activations
	};
	typedef struct _mjDynamics mjDynamics;


	// sensor data; use the sensor desciptors in mjInfo to decode
	struct _mjSensor
	{
		int nsensordata;                    // size of sensor data array
		float time;                         // simulation time
		float sensordata[mjMAXSZ];          // sensor data array
	};
	typedef struct _mjSensor mjSensor;


	// body positions and orientations in Cartesian coordinates (from forward kinematics)
	struct _mjBody
	{
		int nbody;                          // number of bodies
		float time;                         // simulation time
		float pos[mjMAXSZ][3];              // positions
		float mat[mjMAXSZ][9];              // frame orientations
	};
	typedef struct _mjBody mjBody;


	// geom positions and orientations in Cartesian coordinates
	struct _mjGeom
	{
		int ngeom;                          // number of geoms
		float time;                         // simulation time
		float pos[mjMAXSZ][3];              // positions
		float mat[mjMAXSZ][9];              // frame orientations
	};
	typedef struct _mjGeom mjGeom;


	// site positions and orientations in Cartesian coordinates
	struct _mjSite
	{
		int nsite;                          // number of sites
		float time;                         // simulation time
		float pos[mjMAXSZ][3];              // positions
		float mat[mjMAXSZ][9];              // frame orientations
	};
	typedef struct _mjSite mjSite;


	// tendon lengths and velocities
	struct _mjTendon
	{
		int ntendon;                        // number of tendons
		float time;                         // simulation time
		float length[mjMAXSZ];              // tendon lengths
		float velocity[mjMAXSZ];            // tendon velocities    
	};
	typedef struct _mjTendon mjTendon;


	// actuator lengths, velocities, and (scalar) forces in actuator space
	struct _mjActuator
	{
		int nu;                             // number of actuators
		float time;                         // simulation time
		float length[mjMAXSZ];              // actuator lengths
		float velocity[mjMAXSZ];            // actuator velocities
		float force[mjMAXSZ];               // actuator forces
	};
	typedef struct _mjActuator mjActuator;


	// generalized forces acting on the system, resulting in dynamics:
	//   M(qpos)*qacc = nonconstraint + constraint
	struct _mjForce
	{
		int nv;                             // number of generalized velocities/forces
		float time;                         // simulation time
		float nonconstraint[mjMAXSZ];       // sum of all non-constraint forces
		float constraint[mjMAXSZ];          // constraint forces (including contacts)
	};
	typedef struct _mjForce mjForce;

	// static information about the model
	struct _mjInfo
	{
		// sizes
		int nq;                             // number of generalized positions
		int nv;                             // number of generalized velocities
		int na;                             // number of actuator activations
		int njnt;                           // number of joints
		int nbody;                          // number of bodies
		int ngeom;                          // number of geoms
		int nsite;                          // number of sites
		int ntendon;                        // number of tendons
		int nu;                             // number of actuators/controls
		int neq;                            // number of equality constraints
		int nkey;                           // number of keyframes
		int nmocap;                         // number of mocap bodies
		int nsensor;                        // number of sensors
		int nsensordata;                    // number of elements in sensor data array
		int nmat;                           // number of materials

											// timing parameters
		float timestep;                     // simulation timestep
		float apirate;                      // API update rate (same as hxRobotInfo.update_rate)

											// sensor descriptors
		int sensor_type[mjMAXSZ];           // sensor type (mjtSensor)
		int sensor_objid[mjMAXSZ];          // id of sensorized object
		int sensor_dim[mjMAXSZ];            // number of (scalar) sensor outputs
		int sensor_adr[mjMAXSZ];            // address in sensor data array

											// joint properties
		int jnt_type[mjMAXSZ];              // joint type (mjtJoint)
		int jnt_bodyid[mjMAXSZ];            // id of body to which joint belongs
		int jnt_qposadr[mjMAXSZ];           // address of joint position data in qpos
		int jnt_dofadr[mjMAXSZ];            // address of joint velocity data in qvel
		float jnt_range[mjMAXSZ][2];        // joint range; (0,0): no limits

											// geom properties
		int geom_type[mjMAXSZ];             // geom type (mjtGeom)
		int geom_bodyid[mjMAXSZ];           // id of body to which geom is attached

											// equality constraint properties
		int eq_type[mjMAXSZ];               // equality constraint type (mjtEq)
		int eq_obj1id[mjMAXSZ];             // id of constrained object
		int eq_obj2id[mjMAXSZ];             // id of 2nd constrained object; -1 if not applicable

											// actuator properties
		int actuator_trntype[mjMAXSZ];      // transmission type (mjtTrn)
		int actuator_trnid[mjMAXSZ][2];     // transmission target id
		float actuator_ctrlrange[mjMAXSZ][2]; // actuator control range; (0,0): no limits
	};
	typedef struct _mjInfo mjInfo;

	//--------------------------- API get/set functions -------------------------------------

	// get dynamic data from simulator
	// get dynamic data from simulator
	//static mjtResult hmj_get_state(mjState* state)
	//{
	//	return mj_get_state(state);
	//}
	//static mjtResult hmj_get_control(mjControl* control)
	//{
	//	return mj_get_control(control);
	//}
	//static mjtResult hmj_get_applied(mjApplied* applied)
	//{
	//	return mj_get_applied(applied);
	//}
	//static mjtResult hmj_get_onebody(mjOneBody* onebody)
	//{
	//	return mj_get_onebody(onebody);
	//}
	//static mjtResult hmj_get_mocap(mjMocap* mocap)
	//{
	//	return mj_get_mocap(mocap);
	//}
	//static mjtResult hmj_get_dynamics(mjDynamics* dynamics)
	//{
	//	return mj_get_dynamics(dynamics);
	//}
	//static mjtResult hmj_get_sensor(mjSensor* sensor)
	//{
	//	return mj_get_sensor(sensor);
	//}
	//static mjtResult hmj_get_body(mjBody* body)
	//{
	//	return mj_get_body(body);
	//}
	//static mjtResult hmj_get_geom(mjGeom* geom)
	//{
	//	return mj_get_geom(geom);
	//}
	//static mjtResult hmj_get_site(mjSite* site)
	//{
	//	return mj_get_site(site);
	//}
	//static mjtResult hmj_get_tendon(mjTendon* tendon)
	//{
	//	return mj_get_tendon(tendon);
	//}
	//static mjtResult hmj_get_actuator(mjActuator* actuator)
	//{
	//	return mj_get_actuator(actuator);
	//}
	//static mjtResult hmj_get_force(mjForce* force)
	//{
	//	return mj_get_force(force);
	//}
	//static mjtResult hmj_get_contact(mjContact* contact)
	//{
	//	return mj_get_contact(contact);
	//}

	//// set dynamic data in simulator
	//static mjtResult hmj_set_state(const mjState* state)
	//{
	//	return mj_set_state(state);
	//}
	//static mjtResult hmj_set_control(const mjControl* control)
	//{
	//	return mj_set_control(control);
	//}
	//static mjtResult hmj_set_applied(const mjApplied* applied)
	//{
	//	return mj_set_applied(applied);
	//}
	//static mjtResult hmj_set_onebody(const mjOneBody* onebody)
	//{
	//	return mj_set_onebody(onebody);
	//}
	//static mjtResult hmj_set_mocap(const mjMocap* mocap)
	//{
	//	return mj_set_mocap(mocap);
	//}

	//// get and set rgba static data in simulator
	////  valid object types: geom, site, tendon, material
	//static mjtResult hmj_get_rgba(const char* type, int id, float* rgba)
	//{
	//	return mj_get_rgba(type, id, rgba);
	//}
	//static mjtResult hmj_set_rgba(const char* type, int id, const float* rgba)
	//{
	//	return mj_set_rgba(type, id, rgba);
	//}


	////--------------------------- API command and information functions ---------------------

	//// connect to simulator
	//static mjtResult hmj_connect(const char* host)
	//{
	//	return mj_connect(host);
	//}

	//// close connection to simulator
	//static mjtResult hmj_close(void)
	//{
	//	return mj_close();
	//}

	//// return last result code
	//static mjtResult hmj_result(void)
	//{
	//	return mj_result();
	//}

	//// return 1 if connected to simulator, 0 otherwise
	//int hmj_connected(void)
	//{
	//	return mj_connected();
	//}

	//// get static properties of current model
	//static mjtResult hmj_info(mjInfo* info)
	//{
	//	return mj_info(info);
	//}

	//// advance simulation if paused, no effect if running
	//static mjtResult hmj_step(void)
	//{
	//	return mj_step();
	//}

	//// set control, step if paused or wait for 1/apirate if running, get sensor data
	//static mjtResult hmj_update(const mjControl* control, mjSensor* sensor)
	//{
	//	return mj_update(control, sensor);
	//}

	//// reset simulation to specified key frame; -1: reset to model reference configuration
	//static mjtResult hmj_reset(int keyframe)
	//{
	//	return mj_reset(keyframe);
	//}

	//// modify state of specified equality constraint (1: enable, 0: disable)
	//static mjtResult hmj_equality(int eqid, int state)
	//{
	//	return mj_equality(eqid, state);
	//}

	//// show text message in simulator; NULL: clear currently shown message
	//static mjtResult hmj_message(const char* message)
	//{
	//	return mj_message(message);
	//}

	//// return id of object with specified type and name; -1: not found; -2: error
	////  valid object types: body, geom, site, joint, tendon, sensor, actuator, equality
	//int hmj_name2id(const char* type, const char* name)
	//{
	//	return mj_name2id(type, name);
	//}

	//// return name of object with specified type and id; NULL: error
	//const char* hmj_id2name(const char* type, int id)
	//{
	//	return mj_id2name(type, id);
	//}

	//// install user error handler
	//void hmj_handler(void(*handler)(int))
	//{
	//	return mj_handler(handler);
	//}