package org.example.processoremulatorfx.Models;

import org.example.processoremulatorfx.*;

import java.util.ArrayList;
import java.util.Arrays;

public class CpuModel implements ICpu {
  private int[] registers = new int[4]; // 4 регистра
  private int[] memory = new int[1024]; // Память (1024 элемента double)
  private String currentInstruction;
  public int[] getRegisters(){ return registers; }
  public int[] getMemory(){ return memory; }

  public String getCurrentInstruction() { return currentInstruction; }

  ArrayList<IObserver> allObserver = new ArrayList<>();
  void eventCall(){
    allObserver.forEach(action->action.event(this));
  }
  public void addObserver(IObserver e){
    allObserver.add(e);
    eventCall();
  }

  public void ClearMemoryAndRegister() {
    currentInstruction = null;
    Arrays.fill(registers, 0);
    Arrays.fill(memory, 0);
    eventCall();
  }

  @Override
  public void exec(Command c) {
    // Реализация команд
    currentInstruction = c.getCommand();
    switch (currentInstruction) {
      case "init":
        int address = Integer.parseInt(c.getArgs(0));
        int value = Integer.parseInt(c.getArgs(1));
        memory[address] = value;
        break;
      case "ld":
        int regIndex = getRegisterIndex(c.getArgs(0));
        address = Integer.parseInt(c.getArgs(1));
        registers[regIndex] = memory[address];
        break;
      case "st":
        regIndex = getRegisterIndex(c.getArgs(0));
        address = Integer.parseInt(c.getArgs(1));
        memory[address] = registers[regIndex];
        break;
      case "mv":
        int sourceRegIndex = getRegisterIndex(c.getArgs(0));
        int destRegIndex = getRegisterIndex(c.getArgs(1));
        registers[sourceRegIndex] = registers[destRegIndex];
        break;
      case "print":
        //printRegisters();
        break;
      case "add":
        registers[3] = registers[0] + registers[1]; // d = a + b
        break;
      case "sub":
        registers[3] = registers[0] - registers[1]; // d = a - b
        break;
      case "mult":
        registers[3] = registers[0] * registers[1]; // d = a * b
        break;
      case "div":
        registers[3] = registers[0] / registers[1]; // d = a / b
        break;
      default:
        throw new IllegalArgumentException("Unknown command: " + c.getCommand());
    }
    eventCall();
  }

  private int getRegisterIndex(String reg) {
    switch (reg) {
      case "a": return 0;
      case "b": return 1;
      case "c": return 2;
      case "d": return 3;
      default: throw new IllegalArgumentException("Unknown register: " + reg);
    }
  }

  public String printRegisters() {
    return registers[0] + " " + registers[1] + " " + registers[2] + " " + registers[3] + "\n";
  }
}
