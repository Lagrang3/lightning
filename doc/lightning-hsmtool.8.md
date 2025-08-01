lightning-hsmtool -- Tool for working with software HSM secrets of lightningd
=============================================================================

SYNOPSIS
--------

```bash
lightning-hsmtool method [ARGUMENTS]...
```

DESCRIPTION
-----------

**lightning-hsmtool** performs various operations on the `hsm_secret`
file used by the software HSM component of **lightningd**.

This can be used to encrypt and decrypt the `hsm_secret` file,
as well as derive secrets used in channel commitments.

METHODS
-------

**encrypt** *hsm\_secret\_path* *password*

  Encrypt the `hsm_secret_path` file so that it can only be decrypted at
**lightningd** startup.
You must give the option **--encrypted-hsm** to **lightningd**.
The password of the `hsm_secret_path` file will be asked whenever you
start **lightningd**.

**decrypt** *hsm\_secret\_path* *password*

  Decrypt the `hsm_secret_path` file that was encrypted with the **encrypt**
method.

**dumpcommitments** *node\_id* *channel\_dbid* *depth* *hsm\_secret\_path* \[*password*\]

  Show the per-commitment secret and point of up to *depth* commitments,
of the specified channel with the specified peer,
identified by the channel database index.
Specify *password* if the `hsm_secret_path` is encrypted.

**guesstoremote** *p2wpkh* *node\_id* *max\_channel\_dbid* *hsm\_secret\_path* \[*password*\]

  Brute-force the private key to our funds from a remote unilateral close
of a channel, in a case where we have lost all database data except for
our `hsm_secret_path`.
The peer must be the one to close the channel (and the funds will remain
unrecoverable until the channel is closed).
*max\_channel\_dbid* is your own guess on what the *channel\_dbid* was,
or at least the maximum possible value,
and is usually no greater than the number of channels that the node has
ever had.
Specify *password* if the `hsm_secret_path` is encrypted.

**derivetoremote** *node\_id* *channel\_dbid* \[*commitment\_point*\] *hsm\_secret*
  Derive the private key to our funds from a remote unilateral close of a channel.
The peer must be the one to close the channel (and the funds will remain
unrecoverable until the channel is closed).
*channel\_dbid* is the identifier of the channel in the CLN database.
If *commitment\_point* is omitted, then the channel is assumed to have been
negotiated with `option_static_remotekey`;
otherwise, *commitment\_point* is the remote per-commitment point.

**generatehsm** *hsm\_secret\_path* \[*lang* *seed\_phrase*  \[*passphrase*\]\]
  Generates a new hsm\_secret using BIP39.  If lang, seed\_phrase and optional passphrase are not provided they will be prompted for.  *lang* can be "en" (English), "es" (Spanish), "fr" (French), "it" ("Italian"), "jp" (Japanese), "zhs" (Chinese Simplified) or "zht" ("Chinese Traditional"). Note that the seed phrase consists of multiple words, so should be surrounded by quotes.

**checkhsm** *hsm\_secret\_path*
  Checks that hsm\_secret matches a BIP39 passphrase.

**dumponchaindescriptors** \[*--show-secrets*\] *hsm\_secret\_path* \[*network*\]
  Dump output descriptors for our onchain wallet.
This command requires the path to the hsm\_secret containing the wallet seed.
If the flag *--show-secrets* is set the command will show the BIP32 extended private
keys, otherwise the extended public keys will be shown.
The descriptors can be used by external services to be able to generate
addresses for our onchain wallet or to spend those funds
provided that the private keys are visible with *--show-secrets*.
The descriptors can be loaded into a bitcoin-core wallet for example,
using the `importmulti` or `importdescriptors` RPC calls.
If the hsm\_secret was encrypted the command will prompt for a decryption
password.
To generate descriptors using testnet master keys, you may specify *testnet* as
the last parameter. By default, mainnet-encoded keys are generated.

**makerune** *hsm\_secret\_path*
  Make a master rune for this node (with `uniqueid` 0)
This produces the same results as lightning-commando-rune(7) on a fresh node.
You will still need to create a rune once the node starts, if you want commando to work (as it is only activated once it has generated one).

**getcodexsecret** *hsm\_secret\_path* *id*
  Print out the BIP-93 formatted HSM secret, for use with `--recover`.  The `id` is any 4 character string you can use to identify this secret (e.g. `ad00`): it cannot contain `i`, `o`, or `b`, but can contain digits except `1`.

**getemergencyrecover** *emergency.recover\_path*
  Print out the bech32 encoded emergency.recover file.

**getnodeid** *hsm\_secret\_path*
  Print out the node id that a node using this hsm secret would have: useful for verifying that you are accessing the correct secret!

BUGS
----

You should report bugs on our github issues page, and maybe submit a fix
to gain our eternal gratitude!

AUTHOR
------

ZmnSCPxj <<ZmnSCPxj@protonmail.com>> wrote the initial version of
this man page, but many others did the hard work of actually implementing
**lightning-hsmtool**.

SEE ALSO
--------

lightningd(8), lightningd-config(5)

RESOURCES
---------

Main web site: <https://github.com/ElementsProject/lightning>

COPYING
-------

Note: the modules in the ccan/ directory have their own licenses, but
the rest of the code is covered by the BSD-style MIT license.
Main web site: <https://github.com/ElementsProject/lightning>
