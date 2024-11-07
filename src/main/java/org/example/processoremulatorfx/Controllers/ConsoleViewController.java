package org.example.processoremulatorfx.Controllers;

import javafx.fxml.FXML;
import javafx.scene.control.Label;
import javafx.scene.layout.VBox;
import javafx.scene.text.Font;
import javafx.scene.text.Text;
import org.example.processoremulatorfx.BModels.BCpuModel;
import org.example.processoremulatorfx.IObserver;
import org.example.processoremulatorfx.Models.CpuModel;
import org.example.processoremulatorfx.Models.ExecuterModel;
import org.example.processoremulatorfx.Models.ProgramModel;

import java.util.Objects;

public class ConsoleViewController implements IObserver {
  CpuModel cpuModel = BCpuModel.build();
  Font monoFont = Font.font("monospaced", 12);
  int iter = 0;

  String textConsole = "";


  @FXML
  void initialize() { cpuModel.addObserver(this); }

  @FXML
  VBox Console;

  @Override
  public void event(CpuModel c) {
    if (Objects.deepEquals(c.getCurrentInstruction(), "print")) {
      iter++;
      Console.getChildren().clear();
      textConsole = iter + ": " + c.printRegisters() + textConsole;
      Text st = new Text(textConsole);
      st.setFont(monoFont);
      Console.getChildren().add(new VBox(st));
    }
  }

  @Override
  public void event(ExecuterModel e) {
  }

  @Override
  public void event(ProgramModel m) {}
}
