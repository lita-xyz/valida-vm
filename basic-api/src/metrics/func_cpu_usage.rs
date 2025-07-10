use core::panic;
use std::{cell::RefCell, fmt::Display, ops::DerefMut, rc::Rc};

use p3_baby_bear::BabyBear;
use valida_cpu::{JalInstruction, JalvInstruction};
use valida_elf::{create_func_address_to_name_map, minimal_parse_elf};
use valida_machine::{Instruction, InstructionWord};

use crate::BasicMachine;

#[derive(Debug)]
struct CallTree<L: Display> {
    num_cycles: i64,
    address: L,
    callees: Vec<Rc<RefCell<CallTree<L>>>>,
}

type CallTreeByAddress = CallTree<u32>;
type CallTreeByFuncName = CallTree<String>;

#[derive(Debug)]
struct CallStack(Vec<Rc<RefCell<CallTreeByAddress>>>);

pub trait MemFpRelativeReader {
    fn get(&self, fp_offset: i32) -> i32;
}

#[derive(Debug)]
pub struct FuncCpuUsage {
    call_tree: Rc<RefCell<CallTreeByAddress>>,
    call_stack: CallStack,
}

fn empty_call_tree(address: u32) -> CallTreeByAddress {
    CallTree {
        num_cycles: 0,
        address,
        callees: Vec::new(),
    }
}

fn convert_call_tree<F>(call_tree: &CallTreeByAddress, converter: &F) -> CallTreeByFuncName
where
    F: Fn(u64) -> String,
{
    CallTreeByFuncName {
        num_cycles: call_tree.num_cycles,
        address: converter(call_tree.address as u64),
        callees: call_tree
            .callees
            .iter()
            .map(|child| Rc::new(RefCell::new(convert_call_tree(&child.borrow(), converter))))
            .collect(),
    }
}

fn convert_to_flamegraph<L: Display>(prefix: String, call_tree: &CallTree<L>) -> Vec<String> {
    let new_prefix = if prefix.is_empty() {
        prefix
    } else {
        prefix + "; "
    };

    let current_node_prefix = format!("{}{}", new_prefix, call_tree.address);
    let current_node_entry = format!("{} {}\n", current_node_prefix, call_tree.num_cycles);

    let mut from_children: Vec<_> = call_tree
        .callees
        .iter()
        .flat_map(|child| convert_to_flamegraph(current_node_prefix.clone(), &child.borrow()))
        .collect();

    from_children.push(current_node_entry);
    from_children
}

impl FuncCpuUsage {
    pub fn initialize() -> FuncCpuUsage {
        let start = Rc::new(RefCell::new(empty_call_tree(0)));

        FuncCpuUsage {
            call_tree: start.clone(),
            call_stack: CallStack(vec![start.clone()]),
        }
    }

    fn increment_num_cycles_of_current_func(&mut self) {
        self.call_stack
            .0
            .last_mut()
            .unwrap()
            .borrow_mut()
            .num_cycles += 1;
    }

    fn call_function(&mut self, address: u32) {
        let call_stack = &mut self.call_stack.0;
        let current_func = call_stack.last().unwrap().clone();

        let option_callee = current_func
            .borrow()
            .callees
            .iter()
            .find(|call_tree| call_tree.borrow().address == address)
            .map(Rc::clone);

        match option_callee {
            Some(callee) => call_stack.push(callee.clone()),
            None => {
                let new_func = Rc::new(RefCell::new(empty_call_tree(address)));

                current_func.borrow_mut().callees.push(new_func.clone());
                call_stack.push(new_func.clone());
            }
        }
    }

    fn return_from_function(&mut self) {
        match self.call_stack.0.pop() {
            Some(_) => {}
            None => panic!("trying to return from _start"),
        }
    }

    pub fn on_step<Reader: MemFpRelativeReader>(
        &mut self,
        inst: &InstructionWord<i32>,
        reader: &Reader,
    ) {
        self.increment_num_cycles_of_current_func();

        match inst.opcode {
            <JalInstruction as Instruction<BasicMachine<BabyBear>, BabyBear>>::OPCODE => {
                // JAL
                let address = inst.operands.0[1] as u32;

                self.call_function(address);
            }
            <JalvInstruction as Instruction<BasicMachine<BabyBear>, BabyBear>>::OPCODE => {
                // JALV
                let b = inst.operands.0[1];
                let c = inst.operands.0[2];

                let address = reader.get(b) as u32;
                let fp_adjustment = reader.get(c);

                if (fp_adjustment == 0) {
                    panic!("JALV with [c] == 0");
                }

                if fp_adjustment < 0 {
                    self.call_function(address);
                } else {
                    self.return_from_function();
                }
            }
            _ => {}
        }
    }

    pub fn as_flamegraph(&self, file: &Vec<u8>) -> String {
        let elf = minimal_parse_elf(file);
        let symbol_names_map = create_func_address_to_name_map(&elf).unwrap();
        let renamed_call_tree = convert_call_tree(&self.call_tree.borrow(), &|address| {
            symbol_names_map
                .get(&address)
                .unwrap_or(&format!("<{:#08x}>", address))
                .clone()
        });

        convert_to_flamegraph("".to_string(), &renamed_call_tree).concat()
    }
}
