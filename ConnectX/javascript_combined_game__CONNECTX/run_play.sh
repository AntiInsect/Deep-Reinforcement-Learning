if [[ $1 = "-t" ]]
then
  python test.py
  exit 0
fi

while true
do
  echo ""
  echo "Next step, please enter the corresponding number to perform the operation."
  echo "---------------------------"
  echo "0: Exit"
  echo "1: Configure the game"
  echo "2: Run Game CLI"
  echo "3: Run visual game by Web server"
  echo "4: Neural network training"
  echo "---------------------------"

  read -p ": " input
  case $input in
  0)
    break
    ;;
  1)
    python configure.py
    continue
    ;;
  2)
    python start_from_console.py
    continue
    ;;
  3)
    python start_from_web.py
    continue
    ;;
  4)
    python train.py
    continue
    ;;
  *)
    echo "The input is incorrect, please try again."
    continue
    ;;
  esac
done
