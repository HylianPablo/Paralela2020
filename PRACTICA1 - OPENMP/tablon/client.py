#!/usr/bin/python2
# -*- coding: utf-8 -*-

################################################################################
#
# Servidor Tablon - Cliente 1.5
# (c)Javier Fresno, 2015
#
# Servidor Tablon - Cliente 1.8
# (c) Arturo Gonzalez-Escribano, 2019
# Generalized for other contest types
#
################################################################################


# Imports
import sys
import binascii
import pickle
import re
import socket
from optparse import OptionParser
import getpass
import hashlib


# REPLACE SRP
  # N    A large safe prime (N = 2q+1, where q is prime)
  #      All arithmetic is done modulo N.
  # g    A generator modulo N
  # k    Multiplier parameter (k = H(N, g) in SRP-6a, k = 3 for legacy SRP-6)
  # s    User's salt
  # I    Username
  # p    Cleartext Password
  # H()  One-way hash function
  # ^    (Modular) Exponentiation
  # u    Random scrambling parameter
  # a,b  Secret ephemeral values
  # A,B  Public ephemeral values
  # x    Private key (derived from p and s)
  # v    Password verifier

import hashlib
import os
import binascii

SHA1   = 0
SHA224 = 1
SHA256 = 2
SHA384 = 3
SHA512 = 4

NG_1024   = 0
NG_2048   = 1
NG_4096   = 2
NG_8192   = 3
NG_CUSTOM = 4

_hash_map = { SHA1   : hashlib.sha1,
              SHA224 : hashlib.sha224,
              SHA256 : hashlib.sha256,
              SHA384 : hashlib.sha384,
              SHA512 : hashlib.sha512 }


_ng_const = (
# 1024-bit
('''\
EEAF0AB9ADB38DD69C33F80AFA8FC5E86072618775FF3C0B9EA2314C9C256576D674DF7496\
EA81D3383B4813D692C6E0E0D5D8E250B98BE48E495C1D6089DAD15DC7D7B46154D6B6CE8E\
F4AD69B15D4982559B297BCF1885C529F566660E57EC68EDBC3C05726CC02FD4CBF4976EAA\
9AFD5138FE8376435B9FC61D2FC0EB06E3''',
"2"),
# 2048
('''\
AC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4\
A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF60\
95179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF\
747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B907\
8717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB37861\
60279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DB\
FBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF73''',
"2"),
# 4096
('''\
FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E08\
8A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B\
302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9\
A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE6\
49286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8\
FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D\
670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C\
180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718\
3995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D\
04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7D\
B3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D226\
1AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200C\
BBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFC\
E0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B26\
99C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB\
04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2\
233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127\
D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C934063199\
FFFFFFFFFFFFFFFF''',
"5"),
# 8192
('''\
FFFFFFFFFFFFFFFFC90FDAA22168C234C4C6628B80DC1CD129024E08\
8A67CC74020BBEA63B139B22514A08798E3404DDEF9519B3CD3A431B\
302B0A6DF25F14374FE1356D6D51C245E485B576625E7EC6F44C42E9\
A637ED6B0BFF5CB6F406B7EDEE386BFB5A899FA5AE9F24117C4B1FE6\
49286651ECE45B3DC2007CB8A163BF0598DA48361C55D39A69163FA8\
FD24CF5F83655D23DCA3AD961C62F356208552BB9ED529077096966D\
670C354E4ABC9804F1746C08CA18217C32905E462E36CE3BE39E772C\
180E86039B2783A2EC07A28FB5C55DF06F4C52C9DE2BCBF695581718\
3995497CEA956AE515D2261898FA051015728E5A8AAAC42DAD33170D\
04507A33A85521ABDF1CBA64ECFB850458DBEF0A8AEA71575D060C7D\
B3970F85A6E1E4C7ABF5AE8CDB0933D71E8C94E04A25619DCEE3D226\
1AD2EE6BF12FFA06D98A0864D87602733EC86A64521F2B18177B200C\
BBE117577A615D6C770988C0BAD946E208E24FA074E5AB3143DB5BFC\
E0FD108E4B82D120A92108011A723C12A787E6D788719A10BDBA5B26\
99C327186AF4E23C1A946834B6150BDA2583E9CA2AD44CE8DBBBC2DB\
04DE8EF92E8EFC141FBECAA6287C59474E6BC05D99B2964FA090C3A2\
233BA186515BE7ED1F612970CEE2D7AFB81BDD762170481CD0069127\
D5B05AA993B4EA988D8FDDC186FFB7DC90A6C08F4DF435C934028492\
36C3FAB4D27C7026C1D4DCB2602646DEC9751E763DBA37BDF8FF9406\
AD9E530EE5DB382F413001AEB06A53ED9027D831179727B0865A8918\
DA3EDBEBCF9B14ED44CE6CBACED4BB1BDB7F1447E6CC254B33205151\
2BD7AF426FB8F401378CD2BF5983CA01C64B92ECF032EA15D1721D03\
F482D7CE6E74FEF6D55E702F46980C82B5A84031900B1C9E59E7C97F\
BEC7E8F323A97A7E36CC88BE0F1D45B7FF585AC54BD407B22B4154AA\
CC8F6D7EBF48E1D814CC5ED20F8037E0A79715EEF29BE32806A1D58B\
B7C5DA76F550AA3D8A1FBFF0EB19CCB1A313D55CDA56C9EC2EF29632\
387FE8D76E3C0468043E8F663F4860EE12BF2D5B0B7474D6E694F91E\
6DBE115974A3926F12FEE5E438777CB6A932DF8CD8BEC4D073B931BA\
3BC832B68D9DD300741FA7BF8AFC47ED2576F6936BA424663AAB639C\
5AE4F5683423B4742BF1C978238F16CBE39D652DE3FDB8BEFC848AD9\
22222E04A4037C0713EB57A81A23F0C73473FC646CEA306B4BCBC886\
2F8385DDFA9D4B7FA2C087E879683303ED5BDD3A062B3CF5B3A278A6\
6D2A13F83F44F82DDF310EE074AB6A364597E899A0255DC164F31CC5\
0846851DF9AB48195DED7EA1B1D510BD7EE74D73FAF36BC31ECFA268\
359046F4EB879F924009438B481C6CD7889A002ED5EE382BC9190DA6\
FC026E479558E4475677E9AA9E3050E2765694DFC81F56E880B96E71\
60C980DD98EDD3DFFFFFFFFFFFFFFFFF''',
'0x13')
)

def get_ng( ng_type, n_hex, g_hex ):
    if ng_type < NG_CUSTOM:
        n_hex, g_hex = _ng_const[ ng_type ]
    return int(n_hex,16), int(g_hex,16)


def bytes_to_long(s):
    n = ord(s[0])
    for b in ( ord(x) for x in s[1:] ):
        n = (n << 8) | b
    return n


def long_to_bytes(n):
    l = list()
    x = 0
    off = 0
    while x != n:
        b = (n >> off) & 0xFF
        l.append( chr(b) )
        x = x | (b << off)
        off += 8
    l.reverse()
    return ''.join(l)


def get_random( nbytes ):
    return bytes_to_long( os.urandom( nbytes ) )


def get_random_of_length( nbytes ):
    offset = (nbytes*8) - 1
    return get_random( nbytes ) | (1 << offset)


def old_H( hash_class, s1, s2 = '', s3=''):
    if isinstance(s1, (long, int)):
        s1 = long_to_bytes(s1)
    if s2 and isinstance(s2, (long, int)):
        s2 = long_to_bytes(s2)
    if s3 and isinstance(s3, (long, int)):
        s3 = long_to_bytes(s3)
    s = s1 + s2 + s3
    return long(hash_class(s).hexdigest(), 16)


def H( hash_class, *args, **kwargs ):
    h = hash_class()

    for s in args:
        if s is not None:
            h.update( long_to_bytes(s) if isinstance(s, (long, int)) else s )

    return long( h.hexdigest(), 16 )



#N = 0xAC6BDB41324A9A9BF166DE5E1389582FAF72B6651987EE07FC3192943DB56050A37329CBB4A099ED8193E0757767A13DD52312AB4B03310DCD7F48A9DA04FD50E8083969EDB767B0CF6095179A163AB3661A05FBD5FAAAE82918A9962F0B93B855F97993EC975EEAA80D740ADBF4FF747359D041D5C33EA71D281E446B14773BCA97B43A23FB801676BD207A436C6481F1D2B9078717461A5B9D32E688F87748544523B524B0D57D5EA77A2775D2ECFA032CFBDBF52FB3786160279004E57AE6AF874E7303CE53299CCC041C7BC308D82A5698F3A8D0C38271AE35F8E9DBFBB694B5C803D89F7AE435DE236D525F54759B65E372FCD68EF20FA7111F9E4AFF73;
#g = 2;
#k = H(N,g)

def HNxorg( hash_class, N, g ):
    hN = hash_class( long_to_bytes(N) ).digest()
    hg = hash_class( long_to_bytes(g) ).digest()

    return ''.join( chr( ord(hN[i]) ^ ord(hg[i]) ) for i in range(0,len(hN)) )



def gen_x( hash_class, salt, username, password ):
    return H( hash_class, salt, H( hash_class, username + ':' + password ) )




def create_salted_verification_key( username, password, hash_alg=SHA1, ng_type=NG_2048, n_hex=None, g_hex=None ):
    if ng_type == NG_CUSTOM and (n_hex is None or g_hex is None):
        raise ValueError("Both n_hex and g_hex are required when ng_type = NG_CUSTOM")
    hash_class = _hash_map[ hash_alg ]
    N,g = get_ng( ng_type, n_hex, g_hex )
    _s = long_to_bytes( get_random( 4 ) )
    _v = long_to_bytes( pow(g,  gen_x( hash_class, _s, username, password ), N) )

    return _s, _v



def calculate_M( hash_class, N, g, I, s, A, B, K ):
    h = hash_class()
    h.update( HNxorg( hash_class, N, g ) )
    h.update( hash_class(I).digest() )
    h.update( long_to_bytes(s) )
    h.update( long_to_bytes(A) )
    h.update( long_to_bytes(B) )
    h.update( K )
    return h.digest()


def calculate_H_AMK( hash_class, A, M, K ):
    h = hash_class()
    h.update( long_to_bytes(A) )
    h.update( M )
    h.update( K )
    return h.digest()




class Verifier (object):

    def __init__(self, username, bytes_s, bytes_v, bytes_A, hash_alg=SHA1, ng_type=NG_2048, n_hex=None, g_hex=None, bytes_b=None):
        if ng_type == NG_CUSTOM and (n_hex is None or g_hex is None):
            raise ValueError("Both n_hex and g_hex are required when ng_type = NG_CUSTOM")
        if bytes_b and len(bytes_b) != 32:
            raise ValueError("32 bytes required for bytes_b")
        self.s = bytes_to_long(bytes_s)
        self.v = bytes_to_long(bytes_v)
        self.I = username
        self.K = None
        self._authenticated = False

        N,g        = get_ng( ng_type, n_hex, g_hex )
        hash_class = _hash_map[ hash_alg ]
        k          = H( hash_class, N, g )

        self.hash_class = hash_class
        self.N          = N
        self.g          = g
        self.k          = k

        self.A = bytes_to_long(bytes_A)

        # SRP-6a safety check
        self.safety_failed = self.A % N == 0

        if not self.safety_failed:

            if bytes_b:
                self.b = bytes_to_long(bytes_b)
            else:
                self.b = get_random_of_length( 32 )
            self.B = (k*self.v + pow(g, self.b, N)) % N
            self.u = H(hash_class, self.A, self.B)
            self.S = pow(self.A*pow(self.v, self.u, N ), self.b, N)
            self.K = hash_class( long_to_bytes(self.S) ).digest()
            self.M = calculate_M( hash_class, N, g, self.I, self.s, self.A, self.B, self.K )
            self.H_AMK = calculate_H_AMK( hash_class, self.A, self.M, self.K )


    def authenticated(self):
        return self._authenticated


    def get_username(self):
        return self.I


    def get_ephemeral_secret(self):
        return long_to_bytes(self.b)


    def get_session_key(self):
        return self.K if self._authenticated else None

    # returns (bytes_s, bytes_B) on success, (None,None) if SRP-6a safety check fails
    def get_challenge(self):
        if self.safety_failed:
            return None,None
        else:
            return (long_to_bytes(self.s), long_to_bytes(self.B))

    # returns H_AMK on success, None on failure
    def verify_session(self, user_M):
        if not self.safety_failed and user_M == self.M:
            self._authenticated = True
            return self.H_AMK




class User (object):
    def __init__(self, username, password, hash_alg=SHA1, ng_type=NG_2048, n_hex=None, g_hex=None, bytes_a=None):
        if ng_type == NG_CUSTOM and (n_hex is None or g_hex is None):
            raise ValueError("Both n_hex and g_hex are required when ng_type = NG_CUSTOM")
        if bytes_a and len(bytes_a) != 32:
            raise ValueError("32 bytes required for bytes_a")
        N,g        = get_ng( ng_type, n_hex, g_hex )
        hash_class = _hash_map[ hash_alg ]
        k          = H( hash_class, N, g )

        self.I     = username
        self.p     = password
        if bytes_a:
            self.a = bytes_to_long(bytes_a)
        else:
            self.a = get_random_of_length( 32 )
        self.A     = pow(g, self.a, N)
        self.v     = None
        self.M     = None
        self.K     = None
        self.H_AMK = None
        self._authenticated = False

        self.hash_class = hash_class
        self.N          = N
        self.g          = g
        self.k          = k


    def authenticated(self):
        return self._authenticated


    def get_username(self):
        return self.I


    def get_ephemeral_secret(self):
        return long_to_bytes(self.a)


    def get_session_key(self):
        return self.K if self._authenticated else None


    def start_authentication(self):
        return (self.I, long_to_bytes(self.A))


    # Returns M or None if SRP-6a safety check is violated
    def process_challenge(self, bytes_s, bytes_B):

        self.s = bytes_to_long( bytes_s )
        self.B = bytes_to_long( bytes_B )

        N = self.N
        g = self.g
        k = self.k

        hash_class = self.hash_class

        # SRP-6a safety check
        if (self.B % N) == 0:
            return None

        self.u = H( hash_class, self.A, self.B )

        # SRP-6a safety check
        if self.u == 0:
            return None

        self.x = gen_x( hash_class, self.s, self.I, self.p )

        self.v = pow(g, self.x, N)

        self.S = pow((self.B - k*self.v), (self.a + self.u*self.x), N)

        self.K     = hash_class( long_to_bytes(self.S) ).digest()
        self.M     = calculate_M( hash_class, N, g, self.I, self.s, self.A, self.B, self.K )
        self.H_AMK = calculate_H_AMK(hash_class, self.A, self.M, self.K)

        return self.M


    def verify_session(self, host_HAMK):
        if self.H_AMK == host_HAMK:
            self._authenticated = True

# REPLACE SRP


# CLIENT VERSION
CLIENT_VERSION = '1.8'

DEFAULT_PORT = 8080
OTHER_PORT = 8080


# MAX SIZE: This is checked again in the server
MAX_PROGRAM_SIZE = 15000 * 10 # This is 10 times the size of mpi version reference version (curso 1314)

#
# Class for the Secure session
#
class ClientSession():

	# Constructor
	def __init__(self):

		self.connected = False
		self.error = None

	# Connect and authenticate
	def connect(self,host,port,user,password):


		print ('Conecting ...')

		# 1. Connect the socket
		try:
			self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

			# Try to connect either to the given port or to other (80)
			try:
				self.socket.connect((host, port))
				print ('Connected to %s@%s:%d' % (user,host,port))
			except:
				if not OTHER_PORT:
					raise
				port = OTHER_PORT
				self.socket.connect((host, port))
				print ('Connected to %s@%s:%d' % (user,host,port))

			self.wfile = self.socket.makefile('w+',bufsize=0)
			self.rfile = self.socket.makefile('r+')

		except socket.error:
			self.error = "Error connecting to server"
			return


		# 2. Try to authenticate
		try:

			# Send start message
			self.wfile.write('CONNECT /launcher HTTP/1.0\n\n')

			# Send version
			self.wfile.write(CLIENT_VERSION + '\n')

			# Get server ok
			ok = self.rfile.readline().strip()
			if ok != 'OK':
				print (ok)
				sys.exit(-1)

			# Authenticate user
			usr = User(user, password, hash_alg=SHA256)
			uname, A = usr.start_authentication()
			self.wfile.write(uname+'\n')
			self.wfile.write(binascii.b2a_hex(A)+'\n')

			# Get the challenge
			s = binascii.a2b_hex(self.rfile.readline().strip())
			B = binascii.a2b_hex(self.rfile.readline().strip())

			# Process challenge	and send response
			M = usr.process_challenge(s, B)
			if M is None:
				raise Exception()
			self.wfile.write(binascii.b2a_hex(M)+'\n')

			# Get the session
			HAMK = binascii.a2b_hex(self.rfile.readline().strip())
			usr.verify_session(HAMK)

			# Check authentication
			if usr.authenticated():
				print ("Successful authentication")
				self.connected = True
			else:
				raise Exception()

			# Get the session key
			self.key = usr.get_session_key()

		except:
			self.error = "Error authentication to server failed"
			return	

	# Send the request
	def send(self,request):

		# Msg
		print("Sending request")

		# Marshalling and send of the request
		data = pickle.dumps(request)
		self.wfile.write(str(len(data))+'\n')
		self.wfile.write(data)

		# Compute the hash of the file and our key
		m = hashlib.sha256()
		m.update(data)
		m.update(self.key)
		self.wfile.write(m.hexdigest()+'\n')

		# Get the returned message
		return self.rfile.read()
		self.socket.close()

	# Close the socket
	def close(self):
		self.socket.close()






# Class that contains a request for the execution of a program
class ExecutionRequest():

	# Constructor
	def __init__(self,options):

		# Get the needed options
		self.nprocs = options.nprocs
		self.threads = options.threads
		self.program = base=os.path.basename(options.program)
		self.arguments = options.args
		self.queue = options.queue
		if options.reference != None:
			self.reference = True
			self.reference_name = options.reference
		else:
			self.reference = False
		self.timewall = options.timewall
		self.cl1 = options.cl1

		# Read the file
		with file(options.program) as f:
			self.content = f.read()

		# Copy kernels (if any)
		self.copy_kernels(options.program)
		
		if len(self.content) > MAX_PROGRAM_SIZE:
			print ("Program too big")
			sys.exit(-1)


	# Search for all cuda kernels
	def copy_kernels(self,programpath):

		regex = re.compile('^\s*[#]include\s*["](kernel.*[.]cu)["]',re.MULTILINE)
		includes = re.finditer(regex, self.content)

		# Iterate over the kernels
		for include in includes:
			self.copy_kernel(include.group(0),include.group(1),programpath)




	# Search for a cuda kernel and inserts it on the content
	def copy_kernel(self,kernelinclude,kernelname,programpath):

		print ("Including kernel %s" % kernelname)

		# Check if there is a kernel.cu in the same folder
		if os.path.dirname(programpath) == '':
			kfile = kernelname
		else:
			kfile = os.path.dirname(programpath) + '/' + kernelname
		if not os.path.exists(kfile):
			print ("Kernel %s not found" % kernelname)
			sys.exit()

		# Read the kernel
		with file(kfile) as f:
			kcontent = f.read()

		# Add comments
		kcontent = '//--------- KERNEL ---------\n' + kcontent + '//--------- KERNEL ---------\n'

		# Escape the content in the kernel.cu
		# kcontent = kcontent.encode("string_escape")
		self.content = self.content.replace(kernelinclude,kcontent)







# Parse the input arguments
###############################################################################

parser = OptionParser(usage='usage: %prog [options] program.c/.cu/.cpp/.asm [program args]')

# Default
parser.set_defaults(port=DEFAULT_PORT)
parser.set_defaults(server='frontendv.infor.uva.es')
parser.set_defaults(nprocs=1)
parser.set_defaults(threads=1)
parser.set_defaults(reference=None)
parser.set_defaults(timewall=None)
parser.set_defaults(cl1=None)

# Group server
parser.add_option('-u','--user')
parser.add_option('-x','--password')
parser.add_option('-s','--server')
parser.add_option('-p','--port',type="int")
parser.add_option('-n','--nprocs',type="int")
parser.add_option('-t','--threads',type="int")
parser.add_option('-q','--queue')
parser.add_option('-r','--reference')
parser.add_option('-w','--timewall',type="int",help='Timewall in seconds')
parser.add_option('-c','--cl1',help='Flag para gestion cache/shared')

(options, args) = parser.parse_args()

if len(args) == 0:
	parser.print_help()
	sys.exit(-1)

if not options.user:
	print ("Missing user name")
	sys.exit(0)

if not options.password:
	options.password = getpass.getpass('Write your password: ')

if options.nprocs < 1:
	print ("Wrong number of processes")
	sys.exit(-1)

if options.threads < 1:
	print ("Wrong number of threads")
	sys.exit(-1)



options.program  = args[0]
options.args = ' '.join(args[1:])


if not options.program.endswith('.c') and not options.program.endswith('.cu') and not options.program.endswith('.cpp') and not options.program.endswith('.asm'):
	parser.print_help()
	print ('Program must end with .c, .cpp, .cu or .asm')
	sys.exit(-1)

# Check file
if not os.path.exists(options.program):
	parser.print_help()
	print ('Program do not exists')
	sys.exit(-1)



# Create the session
cs = ClientSession()
cs.connect(options.server,options.port,options.user,options.password)

# Exit if not connected
if not cs.connected:
	print (cs.error)
	sys.exit(-1)

# Create the request with the options
er = ExecutionRequest(options)

# Send the request and get the sever msg
msg = cs.send(er)
print (msg)



