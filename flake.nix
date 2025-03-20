{
  description = "A Nix-flake-based C/C++ development environment";
  inputs.nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1.*.tar.gz";
  outputs = { self, nixpkgs }:
    let
      supportedSystems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forEachSupportedSystem = f: nixpkgs.lib.genAttrs supportedSystems (system: f {
        pkgs = import nixpkgs { inherit system; config.allowUnfree = true; };
      });
    in
    {
      devShells = forEachSupportedSystem ({ pkgs }:
        let
          boostWithMPI = pkgs.boost.override {
            useMpi = true;
            mpi = pkgs.openmpi;
          };
        in
        {
          default = pkgs.mkShell.override
            {
              # Override stdenv in order to change compiler:
              # stdenv = pkgs.clangStdenv;
            }
            {
              packages = with pkgs; [
                pkg-config
                libtorch-bin
                openmpi
                boostWithMPI  # Use the custom Boost with MPI instead of the default boost
                clang-tools
                cmake
                codespell
                conan
                cppcheck
                doxygen
                gtest
                lcov
                vcpkg
                vcpkg-tool
                valgrind
                perf-tools
                pkgs.linuxPackages_latest.perf
                python3
                python3Packages.scipy
                python3Packages.ipykernel
                python3Packages.pip
                python3Packages.matplotlib
                python3Packages.numpy
                pvs-studio
              ] ++ (if system == "aarch64-darwin" then [ ] else [ gdb ]);
            };
        });
    };
}