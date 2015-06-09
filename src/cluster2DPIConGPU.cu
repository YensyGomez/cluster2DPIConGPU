#include <iostream>
#include <cstdlib>
#include <cmath>
#include <cstdio>
#include <complex>
#include <ctime>
#include <cstring>
#include <fstream>
#include <complex.h>
#include <fftw3.h>
//#include<cufftw.h>
#include <curand_kernel.h>
#define __ ios_base::sync_with_stdio(false);

using namespace std;
//Magnitudes físicas

#define m_e         9.10938291e-31  // Masa del Electrón
#define e           1.6021e-19      // Carga del Electrón
#define k_b         1.3806504e-23   // Constante de Boltzmann
#define epsilon_0	8.854187e-12    // Permitividad eléctrica del vacío

#define max_SPe	10000           // Limite (computacional) de Superpartículas electrónicas
#define max_SPi	10000           // Limite (computacional) de Superpartículas iónicas
#define J_X	64
#define J_Y	16


int electrones=0;
int Iones = 1;
int X=0, Y=1;

//************************
// Parámetros del sistema
//************************

double  razon_masas = 1.98e5;     // m_i/m_e (Plata)
double  m_i = razon_masas*m_e;    // masa Ión
double  flujo_inicial = 4.5e33;   // Flujo inicial (# part/m^2*s)
double  vflux_i_x = 1e3;  // Componentes de Velocidad de flujo iónico
double  vflux_i_y = 1e3;
double  vflux_i_magnitud = sqrt(vflux_i_x*vflux_i_x+vflux_i_y*vflux_i_y); // Velocidad de flujo iónico (m/s) = sqrt(2*k_b*Te/(M_PI*m_i))
double  vflux_e_x = sqrt(razon_masas)*vflux_i_x;
double  vflux_e_y = sqrt(razon_masas)*vflux_i_y;
double  vflux_e_magnitud = sqrt(vflux_e_x*vflux_e_x+vflux_e_y*vflux_e_y);
double  ni03D = flujo_inicial/vflux_i_magnitud; // Concentración inicial de iones (3D)
double  ne03D=flujo_inicial/vflux_e_x; // Concentración inicial de electrones (3D)
double  Te=M_PI*0.5*m_e*pow(vflux_e_x,2)/k_b;   // Temperatura electrónica inicial (°K)
                                                // (vflujo=sqrt(2k_bTe/(pi*me)
double  lambda_D = sqrt(epsilon_0*k_b*Te/(ne03D*pow(e,2)));  //Longitud de Debye
double  om_p = vflux_e_x/lambda_D;                    //Frecuencia del plasma
double  ND = ne03D*pow(lambda_D,3);                          //Parámetro del plasma
int     NTe = 1e5;
int     NTI = 1e5;                                  //Número de partículas "reales"

//***************************
//Constantes de normalización
//***************************

double  t0=1e-13;                   //Escala de tiempo: Tiempo de vaporización
double  x0=vflux_i_x*t0;   //Escala de longitud: Distancia recorrida en x por un ión en el tiempo t_0
//double  x0=lambda_D;                //Escala de longitud: Longitud de Debye
double  n0=double(NTe)/(x0*x0*x0);


//************************
//Parámetros de simulación
//************************

double  delta_X=lambda_D;   //Paso espacial
double  Lmax_x = (J_X-1)*delta_X;
double  Lmax_y = (J_Y-1)*delta_X;
//double  Lmax[2]={(J_X-1)*delta_X, (J_Y-1)*delta_X}; //Tamaño del espacio de simulación.
int     Factor_carga_e=10, Factor_carga_i=10;       //Número de partículas por superpartícula.
int     k_max_inj;   //Tiempo máximo de inyección
int     K_total;     //Tiempo total
int     Ntv=8;
int     le=0, li=0,kt;
int     NTSPe, NTSPI, max_SPe_dt, max_SPi_dt;
double  L_max_x, L_max_y, vphi_i_x, vphi_i_y, vphi_e_x, vphi_e_y,
        fi_Maxwell_x, fi_Maxwell_y, fe_Maxwell_x, fe_Maxwell_y;

//double  L_max[2], vphi_i[2],vphi_e[2], fi_Maxwell[2],fe_Maxwell[2];
double  T,dt,t_0, ne0_3D,ni0_3D,ni0_1D,ne0_1D;
double  vphi_i_magnitud,vphi_e_magnitud ,vpart,x_0,phi_inic;
double  cte_rho=pow(e*t0,2)/(m_i*epsilon_0*pow(x0,3)); //Normalización de epsilon_0
double  phi0=2.*k_b*Te/(M_PI*e), E0=phi0/x0;
//double  cte_E=t0*e*E0/(vflux_i[X]*m_e),fact_el=-1, fact_i=1./razon_masas;
double  cte_E=razon_masas*x0/(vflux_i_x*t0),fact_el=-1, fact_i=1./razon_masas;


int     total_e_perdidos=0;
int     total_i_perdidos=0;
double  mv2perdidas=0;

FILE    *outPot19,*outEnergia, *outPot0_6, *outPot0_9, *outPot1_5, *outPot3_5, *outPot5_5, *outPot15;
FILE    *outFase_ele[81];
FILE    *outFase_ion[81];
FILE    * outPoisson;

double hx;




/*****************************************************************************/
//Creación de kernels para el funcionamiento de la incialización de los datos //
// de manera aleatoria
/*****************************************************************************/

// función de distribución de las coordenadas en x.
__device__  double create_Velocities_X(double fmax, double vphi, double  aleatorio, curandState *states) // función para generar distribución semi-maxwelliana de velocidades de las particulas
                                             // (Ver pág. 291 Computational Physics Fitzpatrick: Distribution functions--> Rejection Method)
{
  double sigma=vphi;                           // sigma=vflujo=vth    ( "dispersión" de la distribución Maxweliana)
  double vmin= 0. ;                            // Rapidez mínima
  double vmax= 4.*sigma;                       // Rapidez máxima
  double v,f,f_random;

  int Idx = blockIdx.x * blockDim.x + threadIdx.x;

  while (true) {

  v=vmin+(vmax-vmin)*aleatorio; // Calcular valor aleatorio de v uniformemente distribuido en el rango [vmin,vmax]
  f =fmax*exp(-(1.0/M_PI)*pow(v/vphi,2));     //
  f_random = fmax*aleatorio;    // Calcular valor aleatorio de f uniformemente distribuido en el rango [0,fmax]

  if (f_random > f)

	  aleatorio = curand_uniform(states + Idx);
  else
	  return  v;
  }
}

// Funcion de distribución para la coordenadas en y.

__device__  double create_Velocities_Y(double fmax1, double vphi1, double  aleatorio, curandState *states) // función para generar distribución semi-maxwelliana de velocidades de las particulas
                                             // (Ver pág. 291 Computational Physics Fitzpatrick: Distribution functions--> Rejection Method)
{
  double sigma=vphi1;                           // sigma=vflujo=vth    ( "dispersión" de la distribución Maxweliana)
  double vmin= -3.*sigma;                            // Rapidez mínima
  double vmax=  3.*sigma;                       // Rapidez máxima
  double v,f,f_random;

  int Idx = blockIdx.x * blockDim.x + threadIdx.x;

  while (true) {

  v=vmin+(vmax-vmin)*aleatorio; // Calcular valor aleatorio de v uniformemente distribuido en el rango [vmin,vmax]
  f =fmax1*exp(-(1.0/M_PI)*pow(v/vphi1,2));     //
  f_random = fmax1*aleatorio;    // Calcular valor aleatorio de f uniformemente distribuido en el rango [0,fmax]

  if (f_random > f)

	  aleatorio = curand_uniform(states + Idx);
  else
	  return  v;
  }
}


__global__  void distribucionVelocidadX(double *pos, double *vel, double fmax, double vphi, curandState *states, int seed, int l){

	int Idx = blockIdx.x * blockDim.x + threadIdx.x;

	seed = (unsigned int) (clock() * Idx);
	curand_init(seed, 0, 0, states + Idx);

	if (Idx < max_SPe) {
		pos[Idx+l] = 0;
		vel[Idx+l] = create_Velocities_X(fmax, vphi, curand_uniform(states + Idx), states); // Distribucion_X

	}

}

__global__  void distribucionVelocidadY(double *pos, double *vel1, double fmax1, double vphi1, curandState *states, int seed, int l, double lmax){

	int Idx = blockIdx.x * blockDim.x + threadIdx.x;

	seed = (unsigned int) (clock() * Idx);
	curand_init(seed, 0, 0, states + Idx);

	if (Idx < max_SPe) {
		pos[Idx+l] = lmax/2.0;
		vel1[Idx+l] = create_Velocities_Y(fmax1, vphi1, curand_uniform(states + Idx), states); // Distribucion_X
	}

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////


int main(void) {



	int size = max_SPe*sizeof(double);


	double *pos_e_x, *pos_e_y, *pos_i_x, *pos_i_y; // vectores posiciones en x and y.
	double *vel_e_x, *vel_e_y, *vel_i_x, *vel_i_y; // vectores velocidades en x and y.


	pos_e_x = (double *) malloc(size);
	pos_e_y = (double *) malloc(size);
	pos_i_x = (double *) malloc(size);
	pos_i_y = (double *) malloc(size);
	vel_e_x = (double *) malloc(size);
	vel_e_y = (double *) malloc(size);
	vel_i_x = (double *) malloc(size);
	vel_i_y = (double *) malloc(size);



	//Declaración de variables para el dispositivo//

	double *pos_e_x_d, *pos_e_y_d, *pos_i_x_d, *pos_i_y_d;
	double *vel_e_x_d, *vel_e_y_d, *vel_i_x_d, *vel_i_y_d;

	cudaMalloc((void **) &pos_e_x_d, size);
	cudaMalloc((void **) &pos_e_y_d, size);
	cudaMalloc((void **) &pos_i_x_d, size);
	cudaMalloc((void **) &pos_i_y_d, size);
	cudaMalloc((void **) &vel_e_x_d, size);
	cudaMalloc((void **) &vel_e_y_d, size);
	cudaMalloc((void **) &vel_i_x_d, size);
	cudaMalloc((void **) &vel_i_y_d, size);

	// crear la semilla para generar el número aleatorio
	curandState *devStates;
	cudaMalloc((void **) &devStates, max_SPe * sizeof(curandState));
	int seed = time(NULL);

	//////////////////////////////////////////////////////////////////////
	//***************************/
	// Normalización de variables
	//***************************/

	L_max_x=Lmax_x/x0;                      // Longitud región de simulación
	L_max_y=Lmax_y/x0;                      // Longitud región de simulación
	t_0=1;
	x_0=1;
	hx=delta_X/x0;                            // Paso espacial
	//double n_0=n0*x0*x0*x0;                   // Densidad de partículas
	dt=1.e-5;                                 // Paso temporal
	ni0_3D=ni03D*pow(x0,3);                   // Concentración de iones inicial 3D
	ne0_3D=ne03D*pow(x0,3);                   // Concentración de electrones inicial 3D
	vphi_i_x=vflux_i_x/vflux_i_x;    // Velocidad térmica Iónica (X)
	vphi_e_x=vflux_e_x/vflux_i_x;    // Velocidad térmica Electrónica (X)
	vphi_i_y=vflux_i_y/vflux_i_x;    // Velocidad térmica Iónica (Y)
	vphi_e_y=vflux_e_y/vflux_i_x;    // Velocidad térmica Electrónica (Y)
	fi_Maxwell_x=  (2./(M_PI*vphi_i_x));    // Valor Máximo de la función de distribución Semi-Maxwelliana Iónica (X)
	fe_Maxwell_x=  (2./(M_PI*vphi_e_x));    // Valor Máximo de la función de distribución Semi-Maxwelliana elecrónica
	fi_Maxwell_y=  (1./(M_PI*vphi_i_y));    // Valor Máximo de la función de distribución Semi-Maxwelliana Iónica
	fe_Maxwell_y=  (1./(M_PI*vphi_e_y));    // Valor Máximo de la función de distribución Semi-Maxwelliana electrónica
	NTSPe=NTe/Factor_carga_e;
	NTSPI=NTI/Factor_carga_i; // Número total de superpartículas


	printf("x0^3=%e \nn0i=%e \nlambda/hx=%e \nTemp = %e\n", x0*x0*x0, ni03D, lambda_D/x0,k_b*Te);

	int Kemision=20;  //Pasos para liberar partículas

	double dt_emision=Kemision*dt; //Tiempo para liberar partículas

	max_SPe_dt=NTSPe*dt_emision;   //Número de Superpartículas el. liberadas cada vez.
	max_SPi_dt=max_SPe_dt;

	//Cantidad de Hilos a correr en cada bloque.
	float blockSize = 1024;
	dim3 dimBlock(ceil(max_SPe/ blockSize), 1, 1);
	dim3 dimGrid(blockSize, 1, 1);


	distribucionVelocidadX<<<blockSize, dimBlock>>>(pos_e_x_d, vel_e_x_d, fe_Maxwell_x,  vphi_e_x, devStates, seed, le);
	cudaDeviceSynchronize();
	distribucionVelocidadX<<<blockSize, dimBlock>>>(pos_i_x_d, vel_i_x_d, fi_Maxwell_x,  vphi_i_x, devStates, seed, li);
	cudaDeviceSynchronize();
	distribucionVelocidadY<<<blockSize, dimBlock>>>(pos_e_y_d, vel_e_y_d, fe_Maxwell_y,  vphi_e_y, devStates, seed, le, L_max_y);
	cudaDeviceSynchronize();
	distribucionVelocidadY<<<blockSize, dimBlock>>>(pos_i_y_d, vel_i_y_d, fi_Maxwell_y,  vphi_i_y, devStates, seed, li, L_max_y);
	cudaDeviceSynchronize();


	cudaMemcpy(pos_e_x, pos_e_x_d,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(pos_e_y, pos_e_y_d,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(pos_i_x, pos_i_x_d,size,cudaMemcpyDeviceToHost);
	cudaMemcpy(pos_i_y, pos_i_y_d,size,cudaMemcpyDeviceToHost);

	ofstream init;
	 init.open("posicionElectroneseIones");//se escribe un archivo de salida para analizar los datos. la salida corresponde al potencial electrostatico en cada celda conocido como phi.
	  for (int i = 0; i < max_SPe; i++){
				init<<pos_e_x[i]<<" "<<pos_e_y[i]<<" "<<pos_i_x[i]<<" "<<pos_e_y[i]<<" ";
			init<<endl;
		}
	init.close();

	free(pos_e_x);
	free(pos_e_y);
	free(pos_i_x);
	free(pos_i_y);
	free(vel_e_x);
	free(vel_e_y);
	free(vel_i_x);
	free(vel_i_y);

	cudaFree(pos_e_x_d);
	cudaFree(pos_e_y_d);
	cudaFree(pos_i_x_d);
	cudaFree(pos_i_y_d);
	cudaFree(vel_e_x_d);
	cudaFree(vel_e_y_d);
	cudaFree(vel_i_x_d);
	cudaFree(vel_i_y_d);

	return 0;
}

